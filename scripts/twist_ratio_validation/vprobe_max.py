"""Held-out twist probe of a walking checkpoint in its own walking env (diagnostic only).

Rules: twist_ratio/walk_status.txt.  The walking policy (63-wide actor with its
observation normalizer) runs in the forward-lean walking env's play config with
fixed twist commands for 300 steps; score window steps 50..299 while up;
twists (v_x, v_y, w_z) in the HOME-levelled trunk frame.  Pushes: world-frame
velocity kicks every 1.0 s (x +0.30 m/s, lateral 0.15 m/s against the commanded
lateral sign).  Seeds 101-105 only (never 42-44).

usage: vprobe_walk.py CKPT OUT.json [--seeds 101,...] [--reps 3] [--pushes none,p30_15]
"""
import argparse
import hashlib
import json
import math
import time
from copy import deepcopy

import torch
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends

from mjlab_microban.robot.microban_constants import HOME_TRUNK_PITCH_RAD
from mjlab_microban.scripts.teleop_v12_bootstrap_gate import _legacy_model
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import inspect_legacy_velocity_checkpoint
from mjlab_microban.tasks.microban_twist_ratio_mdp import home_levelled_twist
from mjlab_microban.tasks.microban_velocity_env_cfg import make_microban_velocity_env_cfg

p = argparse.ArgumentParser()
p.add_argument("ckpt")
p.add_argument("out")
p.add_argument("--seeds", default="101,102,103,104,105")
p.add_argument("--reps", type=int, default=3)
p.add_argument("--pushes", default="none")
a = p.parse_args()
SEEDS = [int(v) for v in a.seeds.split(",")]
assert not set(SEEDS) & {42, 43, 44}, "gate seeds are reserved"
dev = "cuda:0"
STEPS, SETTLE = 300, 50
SCALE = (0.7, 0.3, 1.5)
DIAG = {}
PURE = {
    "FMAX": (0.7, 0.0, 0.0), "BMAX": (-0.5, 0.0, 0.0), "LMAX": (0.0, 0.3, 0.0),
    "RMAX": (0.0, -0.3, 0.0), "YLMAX": (0.0, 0.0, 1.5), "YRMAX": (0.0, 0.0, -1.5),
}
COMMANDS = {**DIAG, **PURE}

configure_torch_backends(allow_tf32=False, deterministic=True)
sha = hashlib.sha256(open(a.ckpt, "rb").read()).hexdigest()
identity, state = inspect_legacy_velocity_checkpoint(a.ckpt, sha)
policy = _legacy_model().to(dev)
policy.load_state_dict(state, strict=True)
policy.eval()

conds = [c for c in DIAG for _ in range(a.reps)] + [c for c in PURE for _ in range(a.reps)]
N = len(conds)
twist_t = torch.tensor([COMMANDS[c] for c in conds], device=dev)
side_t = torch.tensor([-1.0 if COMMANDS[c][1] < 0 else 1.0 for c in conds], device=dev)
PUSH = {"none": None, "p30_15": (0.30, 0.15)}


def side_push(env, env_ids, push_xy, asset_cfg=None):
    asset = env.scene["robot"]
    vel_w = asset.data.root_link_vel_w[env_ids].clone()
    vel_w[:, 0] += push_xy[0]
    vel_w[:, 1] += -push_xy[1] * side_t[env_ids]
    asset.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)


def set_command(env):
    tw = env.command_manager.get_term("twist")
    tw.vel_command_b.copy_(twist_t)
    for f in ("is_heading_env", "is_standing_env", "is_rotation_env"):
        if hasattr(tw, f):
            getattr(tw, f).fill_(False)
    tw.time_left.fill_(float("inf"))


rows = []
t0 = time.time()
for pk in a.pushes.split(","):
    cfg = make_microban_velocity_env_cfg(play=True)
    cfg.scene.num_envs = N
    cfg.seed = SEEDS[0]
    cfg.auto_reset = True  # a fallen env is masked from then on
    cfg.episode_length_s = (STEPS + 2) * cfg.decimation * cfg.sim.mujoco.timestep
    cfg.commands["twist"].resampling_time_range = (1.0e6, 1.0e6)
    cfg.commands["twist"].debug_vis = False
    if PUSH[pk] is None:
        cfg.events.pop("push_robot", None)
    else:
        training = make_microban_velocity_env_cfg(play=False)
        pe = deepcopy(training.events["push_robot"])
        pe.func = side_push
        pe.interval_range_s = (1.0, 1.0)
        pe.params = {"push_xy": PUSH[pk]}
        cfg.events["push_robot"] = pe
    env = ManagerBasedRlEnv(cfg=cfg, device=dev)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    robot = env.scene["robot"]
    fall_h = float(env.termination_manager.get_term_cfg("fell_over").params["minimum_height"])
    try:
        for seed in SEEDS:
            env.reset(seed=seed)
            set_command(env)
            # The reset observation was built before the fixed command was set.
            obs = TensorDict(env.observation_manager.compute(update_history=True), batch_size=[N])
            fell = torch.zeros(N, dtype=torch.bool, device=dev)
            vsum = torch.zeros(N, 3, device=dev)
            cnt = torch.zeros(N, device=dev)
            for step in range(STEPS):
                set_command(env)
                with torch.inference_mode():
                    act = policy(TensorDict({"actor": obs["actor"]}, batch_size=[N]))
                obs, _r, done, _e = wrapped.step(act)
                nf = done.bool().reshape(-1) | env.termination_manager.get_term("fell_over").bool() | (
                    robot.data.root_link_pos_w[:, 2] < fall_h)
                fell = fell | nf
                if step >= SETTLE:
                    v = home_levelled_twist(env, HOME_TRUNK_PITCH_RAD)
                    m = (~fell).float()
                    vsum += v * m[:, None]
                    cnt += m
            mean = (vsum / cnt.clamp(min=1)[:, None]).tolist()
            for i, c in enumerate(conds):
                rows.append({"push": pk, "seed": seed, "cmd": c, "twist": COMMANDS[c],
                             "targets": "full" if c in DIAG else "off",
                             "fell": bool(fell[i]), "up_steps": int(cnt[i]), "mean": mean[i]})
            print(pk, seed, "done", round(time.time() - t0, 1), flush=True)
    finally:
        wrapped.close()


def ratio_metrics(cmd, v):
    c = [cmd[i] / SCALE[i] for i in range(3)]
    w = [v[i] / SCALE[i] for i in range(3)]
    n = math.sqrt(sum(x * x for x in c))
    if n == 0:
        return {"angle_deg": None, "speed": None}
    u = [x / n for x in c]
    along = sum(w[i] * u[i] for i in range(3))
    vn = math.sqrt(sum(x * x for x in w))
    angle = math.degrees(math.acos(max(-1.0, min(1.0, along / vn)))) if vn > 1e-9 else 180.0
    return {"angle_deg": angle, "speed": min(max(along, 0.0), n) / n}


for r in rows:
    r.update(ratio_metrics(r["twist"], r["mean"]))
json.dump({"ckpt": a.ckpt, "sha256": sha, "iteration": identity.iteration, "seeds": SEEDS,
           "reps": a.reps, "rows": rows, "secs": time.time() - t0}, open(a.out, "w"), indent=1)
print("DONE", round(time.time() - t0, 1))
