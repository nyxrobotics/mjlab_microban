"""Held-out check of a walking checkpoint during training (rules W1-W5).

The walking policy (the 63-wide actor with its observation normalizer) runs in
the walking env's play config with fixed twist commands for 300 steps; the
mean HOME-levelled twist over steps 50..299 while up is compared with the
command.  Six diagonal commands with and without pushes (world-frame kicks
every 1.0 s: +0.30 m/s forward and 0.15 m/s against the commanded lateral
sign), six single-axis commands and standing, three repeats each, on seeds
101-105 -- never the judgment's seed 42.

Rules:

* W4 falls of the diagonal commands without / with pushes
  (config/pipeline.yaml ``walk.check.w4_falls``), none single-axis;
* W5 every single-axis command moves its axis the commanded way by the fixed
  minimum (mjlab_microban/twist_pass_line.py: 0.2 m/s forward 0.08, backward
  0.04, 0.1 m/s lateral 0.02, 0.5 rad/s yaw 0.2) and standing drifts at most
  0.05 m/s, 0.05 m/s, 0.2 rad/s;
* W6 standing still: on the standing command without pushes, at most
  ``walk.check.still_touchdowns_per_s`` foot touchdowns per second (both feet,
  per robot, after the 1 s settle).  The reward counts stepping in place as
  motion on a standing command (its best is both feet still, 2026-10-08), so
  an undisturbed stand re-plants a foot at most now and then.

The signed single-axis responses are recorded with each verdict.

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

from mjlab_microban.twist_pass_line import twist_passes

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


def _home_levelled_twist(data, trunk_pitch: float):
    """(N, 3) twist (v_x, v_y, w_z) in the trunk frame with the HOME lean rotated out."""

    import torch
    from mjlab.utils.lab_api.math import quat_apply_inverse, quat_mul

    quat_w = data.root_link_quat_w
    half = -0.5 * trunk_pitch
    unpitch = torch.tensor((math.cos(half), 0.0, math.sin(half), 0.0), device=quat_w.device,
                           dtype=quat_w.dtype).expand_as(quat_w)
    frame = quat_mul(quat_w, unpitch)
    linear = quat_apply_inverse(frame, data.root_link_lin_vel_w)
    angular = quat_apply_inverse(frame, data.root_link_ang_vel_w)
    return torch.stack((linear[:, 0], linear[:, 1], angular[:, 2]), dim=-1)


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
        per[name] = {"falls": sum(r["fell"] for r in rr), "twist": twist, "mean": mean, "signed": signed}
    return {"falls": sum(r["fell"] for r in selected), "per": per}


def evaluate(rows: list[dict], rules: dict[str, Any]) -> dict[str, Any]:
    """The W4-W6 verdicts and the numbers they were taken on."""

    none = _cell(rows, "none", list(DIAGONAL))
    pushed = _cell(rows, "p30_15", list(DIAGONAL))
    single = _cell(rows, "none", list(SINGLE))
    moving = {k: v for k, v in single["per"].items() if k != "S"}
    still = single["per"]["S"]["mean"]
    still_rows = [r for r in rows if r["push"] == "none" and r["cmd"] == "S" and not r["fell"]]
    still_touchdowns = _mean([r["touchdowns_per_s"] for r in still_rows if "touchdowns_per_s" in r])
    single_signed = {name: single["per"][name]["signed"][axis] for name, axis in SINGLE_AXIS.items()}
    checks = {
        "W4": none["falls"] <= rules["w4_falls"][0] and pushed["falls"] <= rules["w4_falls"][1]
        and single["falls"] == 0,
        "W5": len(moving) == len(SINGLE_AXIS) and all(twist_passes(v["twist"], v["mean"]) for v in moving.values())
        and twist_passes(SINGLE["S"], still),
        "W6": math.isfinite(still_touchdowns) and still_touchdowns <= rules["still_touchdowns_per_s"],
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "falls": [none["falls"], pushed["falls"], single["falls"]],
        "single_signed": single_signed,
        "still": still,
        "still_touchdowns_per_s": still_touchdowns,
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
        feet = env.scene.sensors["feet_ground_contact"]
        fall_height = float(env.termination_manager.get_term_cfg("fell_over").params["minimum_height"])
        try:
            for seed in seeds:
                env.reset(seed=seed)
                set_command(env)
                obs = TensorDict(env.observation_manager.compute(update_history=True), batch_size=[n])
                fell = torch.zeros(n, dtype=torch.bool, device=device)
                total = torch.zeros(n, 3, device=device)
                count = torch.zeros(n, device=device)
                down = feet.data.found.reshape(n, -1)[:, :2] > 0
                touchdowns = torch.zeros(n, device=device)
                for step in range(STEPS):
                    set_command(env)
                    with torch.inference_mode():
                        action = policy(TensorDict({"actor": obs["actor"]}, batch_size=[n]))
                    obs, _reward, done, _extras = wrapped.step(action)
                    fell = fell | done.bool().reshape(-1) | env.termination_manager.get_term("fell_over").bool() \
                        | (robot.data.root_link_pos_w[:, 2] < fall_height)
                    now_down = feet.data.found.reshape(n, -1)[:, :2] > 0
                    if step >= SETTLE:
                        up = (~fell).float()
                        total += _home_levelled_twist(robot.data, HOME_TRUNK_PITCH_RAD) * up[:, None]
                        count += up
                        touchdowns += (now_down & ~down).sum(dim=-1).float() * up
                    down = now_down
                mean = (total / count.clamp(min=1)[:, None]).tolist()
                rate = (touchdowns / (count.clamp(min=1) * env.step_dt)).tolist()
                for i, name in enumerate(conds):
                    rows.append({"push": push, "seed": seed, "cmd": name, "twist": commands[name],
                                 "fell": bool(fell[i]), "up_steps": int(count[i]), "mean": mean[i],
                                 "touchdowns_per_s": rate[i]})
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
