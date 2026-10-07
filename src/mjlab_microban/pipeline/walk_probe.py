"""Held-out check of a walking checkpoint during training (rules W1-W5).

The walking policy (the 63-wide actor with its observation normalizer) runs in
the walking env's play config with fixed twist commands for 300 steps; the
mean HOME-levelled twist over steps 50..299 while up is compared with the
command.  Six diagonal commands with and without pushes (world-frame kicks
every 1.0 s: +0.30 m/s forward and 0.15 m/s against the commanded lateral
sign), six single-axis commands and standing, three repeats each, on seeds
101-105 -- never the judgment's seed 42.  This is the probe of the
twist-ratio validation (exp/twist-ratio-validation) that every walker of that
comparison was measured with.

Rules: the walking reward's own pass line (mjlab_microban/twist_pass_line.py)
on the mean twist of each command over its 15 rollouts, and the falls
(config/pipeline.yaml ``walk.check.w4_falls``):

* W1 every diagonal command without pushes: the reward of the mean twist beats
  standing still;
* W2 the same with pushes;
* W3 every single-axis command: the same;
* W4 falls of the diagonal commands without / with pushes, none single-axis;
* W5 standing: the drift costs less than the smallest checked command.

The ratio angle, the speed fraction and the signed single-axis responses are
recorded with each verdict; they are not pass lines.

usage: python -m mjlab_microban.pipeline.walk_probe CKPT OUT.json [--seeds 101,...]
Prints one ``RESULT {json}`` line (the rule verdicts and their numbers).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from copy import deepcopy
from typing import Any

from mjlab_microban.twist_pass_line import twist_passes, twist_value

SCALE = (0.7, 0.3, 1.5)
DIAGONAL = {
    "A": (0.6, 0.3, 1.2), "M": (0.6, -0.3, -1.2),
    "B": (0.7, 0.3, 1.0), "MB": (0.7, -0.3, -1.0),
    "K": (-0.4, 0.25, 0.9), "MK": (-0.4, -0.25, -0.9),
}
SINGLE = {
    "F": (0.2, 0.0, 0.0), "BK": (-0.2, 0.0, 0.0), "LL": (0.0, 0.1, 0.0),
    "LR": (0.0, -0.1, 0.0), "YL": (0.0, 0.0, 0.5), "YR": (0.0, 0.0, -0.5),
    "S": (0.0, 0.0, 0.0),
}
SINGLE_AXIS = {"F": 0, "BK": 0, "LL": 1, "LR": 1, "YL": 2, "YR": 2}
PUSHES = {"none": None, "p30_15": (0.30, 0.15)}
STEPS, SETTLE = 300, 50


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def ratio_angle(command: tuple[float, ...], motion: list[float]) -> float:
    c = [command[i] / SCALE[i] for i in range(3)]
    w = [motion[i] / SCALE[i] for i in range(3)]
    n = math.sqrt(sum(x * x for x in c))
    norm = math.sqrt(sum(x * x for x in w))
    if n == 0.0:
        return float("nan")
    if norm < 1e-9:
        return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, sum(c[i] * w[i] for i in range(3)) / (n * norm)))))


def speed_fraction(command: tuple[float, ...], motion: list[float]) -> float | None:
    c = [command[i] / SCALE[i] for i in range(3)]
    w = [motion[i] / SCALE[i] for i in range(3)]
    n = math.sqrt(sum(x * x for x in c))
    if n == 0.0:
        return None
    along = sum(w[i] * c[i] / n for i in range(3))
    return min(max(along, 0.0), n) / n


def _cell(rows: list[dict], push: str, commands: list[str]) -> dict[str, Any]:
    selected = [r for r in rows if r["push"] == push and r["cmd"] in commands]
    per = {}
    for name in commands:
        rr = [r for r in selected if r["cmd"] == name]
        if not rr:
            continue
        twist = rr[0]["twist"]
        mean = [_mean([r["mean"][i] for r in rr]) for i in range(3)]
        signed = [mean[i] * (1 if twist[i] > 0 else -1 if twist[i] < 0 else 0) for i in range(3)]
        speeds = [s for s in (speed_fraction(twist, r["mean"]) for r in rr) if s is not None]
        per[name] = {"falls": sum(r["fell"] for r in rr), "twist": twist, "mean": mean, "signed": signed,
                     "angle": ratio_angle(twist, mean), "speed": _mean(speeds)}
    speeds = [s for s in (speed_fraction(r["twist"], r["mean"]) for r in selected) if s is not None]
    return {"falls": sum(r["fell"] for r in selected), "speed": _mean(speeds), "per": per}


def evaluate(rows: list[dict], rules: dict[str, Any]) -> dict[str, Any]:
    """The W1-W5 verdicts and the numbers they were taken on."""

    none = _cell(rows, "none", list(DIAGONAL))
    pushed = _cell(rows, "p30_15", list(DIAGONAL))
    single = _cell(rows, "none", list(SINGLE))
    angle_none = max(v["angle"] for v in none["per"].values())
    angle_push = max(v["angle"] for v in pushed["per"].values())

    def values(cell: dict) -> dict[str, float]:
        return {name: twist_value(v["twist"], v["mean"]) for name, v in cell["per"].items()}

    def passes(cell: dict) -> bool:
        return len(cell["per"]) > 0 and all(twist_passes(v["twist"], v["mean"]) for v in cell["per"].values())

    moving = {"per": {k: v for k, v in single["per"].items() if k != "S"}}
    still = single["per"]["S"]["mean"]
    single_signed = {name: single["per"][name]["signed"][axis] for name, axis in SINGLE_AXIS.items()}
    checks = {
        "W1": passes(none),
        "W2": passes(pushed),
        "W3": passes(moving) and len(moving["per"]) == len(SINGLE_AXIS),
        "W4": none["falls"] <= rules["w4_falls"][0] and pushed["falls"] <= rules["w4_falls"][1]
        and single["falls"] == 0,
        "W5": twist_passes(SINGLE["S"], still),
    }
    value = {"none": values(none), "pushed": values(pushed), "single": values(single)}
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "value": value,
        "worst_value": [min(value["none"].values()), min(value["pushed"].values()),
                        min(v for k, v in value["single"].items() if k != "S")],
        "still_value": value["single"]["S"],
        "angle_deg": [angle_none, angle_push],
        "speed": [none["speed"], pushed["speed"]],
        "falls": [none["falls"], pushed["falls"], single["falls"]],
        "single_signed": single_signed,
        "still": still,
    }


def run(checkpoint: str, seeds: list[int], reps: int) -> list[dict]:
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends
    from tensordict import TensorDict

    from mjlab_microban.robot.microban_constants import HOME_TRUNK_PITCH_RAD
    from mjlab_microban.scripts.teleop_v12_bootstrap_gate import _legacy_model
    from mjlab_microban.tasks.microban_teleop_v12_bootstrap import inspect_legacy_velocity_checkpoint
    from mjlab_microban.tasks.microban_twist_ratio_mdp import home_levelled_twist
    from mjlab_microban.tasks.microban_velocity_env_cfg import make_microban_velocity_env_cfg

    if set(seeds) & {42, 43, 44}:
        raise SystemExit("the judgment's seeds are reserved")
    device = "cuda:0"
    commands = {**DIAGONAL, **SINGLE}
    configure_torch_backends(allow_tf32=False, deterministic=True)
    digest = hashlib.sha256(open(checkpoint, "rb").read()).hexdigest()
    _identity, state = inspect_legacy_velocity_checkpoint(checkpoint, digest)
    policy = _legacy_model().to(device)
    policy.load_state_dict(state, strict=True)
    policy.eval()
    conds = [c for c in DIAGONAL for _ in range(reps)] + [c for c in SINGLE for _ in range(reps)]
    n = len(conds)
    twist_t = torch.tensor([commands[c] for c in conds], device=device)
    side_t = torch.tensor([-1.0 if commands[c][1] < 0 else 1.0 for c in conds], device=device)

    def side_push(env, env_ids, push_xy, asset_cfg=None):
        asset = env.scene["robot"]
        vel_w = asset.data.root_link_vel_w[env_ids].clone()
        vel_w[:, 0] += push_xy[0]
        vel_w[:, 1] += -push_xy[1] * side_t[env_ids]
        asset.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)

    def set_command(env):
        term = env.command_manager.get_term("twist")
        term.vel_command_b.copy_(twist_t)
        for flag in ("is_heading_env", "is_standing_env", "is_rotation_env"):
            if hasattr(term, flag):
                getattr(term, flag).fill_(False)
        term.time_left.fill_(float("inf"))

    rows = []
    for push in PUSHES:
        cfg = make_microban_velocity_env_cfg(play=True)
        cfg.scene.num_envs = n
        cfg.seed = seeds[0]
        cfg.auto_reset = True  # a fallen env is masked from then on
        cfg.episode_length_s = (STEPS + 2) * cfg.decimation * cfg.sim.mujoco.timestep
        cfg.commands["twist"].resampling_time_range = (1.0e6, 1.0e6)
        cfg.commands["twist"].debug_vis = False
        if PUSHES[push] is None:
            cfg.events.pop("push_robot", None)
        else:
            event = deepcopy(make_microban_velocity_env_cfg(play=False).events["push_robot"])
            event.func = side_push
            event.interval_range_s = (1.0, 1.0)
            event.params = {"push_xy": PUSHES[push]}
            cfg.events["push_robot"] = event
        env = ManagerBasedRlEnv(cfg=cfg, device=device)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
        robot = env.scene["robot"]
        fall_height = float(env.termination_manager.get_term_cfg("fell_over").params["minimum_height"])
        try:
            for seed in seeds:
                env.reset(seed=seed)
                set_command(env)
                obs = TensorDict(env.observation_manager.compute(update_history=True), batch_size=[n])
                fell = torch.zeros(n, dtype=torch.bool, device=device)
                total = torch.zeros(n, 3, device=device)
                count = torch.zeros(n, device=device)
                for step in range(STEPS):
                    set_command(env)
                    with torch.inference_mode():
                        action = policy(TensorDict({"actor": obs["actor"]}, batch_size=[n]))
                    obs, _reward, done, _extras = wrapped.step(action)
                    fell = fell | done.bool().reshape(-1) | env.termination_manager.get_term("fell_over").bool() \
                        | (robot.data.root_link_pos_w[:, 2] < fall_height)
                    if step >= SETTLE:
                        up = (~fell).float()
                        total += home_levelled_twist(env, HOME_TRUNK_PITCH_RAD) * up[:, None]
                        count += up
                mean = (total / count.clamp(min=1)[:, None]).tolist()
                for i, name in enumerate(conds):
                    rows.append({"push": push, "seed": seed, "cmd": name, "twist": commands[name],
                                 "fell": bool(fell[i]), "up_steps": int(count[i]), "mean": mean[i]})
        finally:
            wrapped.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--seeds", default="101,102,103,104,105")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--rules", required=True, help="JSON of config/pipeline.yaml walk.check")
    args = parser.parse_args()
    started = time.time()
    rows = run(args.checkpoint, [int(s) for s in args.seeds.split(",")], args.reps)
    result = {"checkpoint": args.checkpoint, **evaluate(rows, json.loads(args.rules)),
              "seconds": time.time() - started}
    with open(args.output, "w") as stream:
        json.dump({**result, "rows": rows}, stream, indent=1)
    print("RESULT " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
