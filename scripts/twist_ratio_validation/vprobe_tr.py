"""Held-out twist-ratio probe (diagnostic only; never a gate, never seed 42-44).

Each env holds one (command, targets) condition for 300 steps under the
evaluator's play config (forced moving HMD, nominal resets).  Score window:
steps 50..299 while the robot is up.  Twists are (v_x, v_y, w_z) in the
HOME-levelled trunk frame.  Pushes are world-frame velocity kicks every 1.0 s
(x +0.30, lateral 0.15 against the commanded lateral sign).

Commands
  diagonal (infeasible or near-infeasible, mirrored pairs):
    A (0.6, 0.3, 1.2)  M (0.6, -0.3, -1.2)  B (0.7, 0.3, 1.0)  MB (0.7, -0.3, -1.0)
    K (-0.4, 0.25, 0.9)  MK (-0.4, -0.25, -0.9)
    targets: full (mixed_forward_left foot+hand targets, mirrored for right-
    lateral commands) and hands_off (feet as full, hands inactive)
  single-axis (targets off): F (0.2,0,0) BK (-0.2,0,0) LL (0,0.1,0) LR (0,-0.1,0)
    YL (0,0,0.5) YR (0,0,-0.5) S (0,0,0)

usage: vprobe_tr.py CKPT OUT.json [--seeds 101,...,105] [--reps 3] [--pushes none,p30_15]
"""
import argparse
import json
import math
import time
from copy import deepcopy
from pathlib import Path

import torch
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends

import mjlab_microban.scripts.evaluate_teleop_v12_tracking as ev
from mjlab_microban.robot.microban_constants import HOME_TRUNK_PITCH_RAD
from mjlab_microban.tasks.mdp import home_levelled_root_ang_vel_b, home_levelled_root_lin_vel_b
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import make_microban_teleop_v12_env_cfg

p = argparse.ArgumentParser()
p.add_argument("ckpt")
p.add_argument("out")
p.add_argument("--seeds", default="101,102,103,104,105")
p.add_argument("--reps", type=int, default=3)
p.add_argument("--pushes", default="none,p30_15")
a = p.parse_args()
SEEDS = [int(v) for v in a.seeds.split(",")]
assert not set(SEEDS) & {42, 43, 44}, "gate seeds are reserved"
dev = "cuda:0"
STEPS, SETTLE = 300, 50
SCALE = (0.7, 0.3, 1.5)

DIAG = {
    "A": (0.6, 0.3, 1.2), "M": (0.6, -0.3, -1.2),
    "B": (0.7, 0.3, 1.0), "MB": (0.7, -0.3, -1.0),
    "K": (-0.4, 0.25, 0.9), "MK": (-0.4, -0.25, -0.9),
}
PURE = {
    "F": (0.2, 0.0, 0.0), "BK": (-0.2, 0.0, 0.0), "LL": (0.0, 0.1, 0.0),
    "LR": (0.0, -0.1, 0.0), "YL": (0.0, 0.0, 0.5), "YR": (0.0, 0.0, -0.5),
    "S": (0.0, 0.0, 0.0),
}
COMMANDS = {**DIAG, **PURE}

configure_torch_backends(allow_tf32=False, deterministic=True)
policy, iteration, infos = ev._load_actor(Path(a.ckpt), device=dev)
base = {s.name: s for s in ev._scenarios(ev.FINAL_PROFILE)}
mfl = base["mixed_forward_left"]
zero = (0.0, 0.0, 0.0)


def mirror(v):
    return (v[0], -v[1], v[2])


def side_of(cmd):
    return -1 if cmd[1] < 0 else 1


def layout(cmd, tk):
    foot, hand = mfl.foot_target, mfl.hand_target
    if side_of(cmd) < 0:
        foot = (zero, mirror(mfl.foot_target[0])) if mfl.foot_target[1] == zero else (
            mirror(mfl.foot_target[1]), mirror(mfl.foot_target[0]))
        hand = (mirror(mfl.hand_target[1]), mirror(mfl.hand_target[0]))
    if tk == "full":
        return foot, hand, (True, True)
    if tk == "hands_off":
        return foot, (zero, zero), (False, False)
    return (zero, zero), (zero, zero), (False, False)


conds = [(c, t) for c in DIAG for t in ("full", "hands_off") for _ in range(a.reps)]
conds += [(c, "off") for c in PURE for _ in range(a.reps)]
N = len(conds)
twist_t = torch.tensor([COMMANDS[c] for c, _ in conds], device=dev)
side_t = torch.tensor([float(side_of(COMMANDS[c])) for c, _ in conds], device=dev)
foot_t = torch.tensor([layout(COMMANDS[c], t)[0] for c, t in conds], device=dev)
hand_t = torch.tensor([layout(COMMANDS[c], t)[1] for c, t in conds], device=dev)
act_t = torch.tensor([layout(COMMANDS[c], t)[2] for c, t in conds], device=dev)
PUSH = {"none": None, "p30_15": (0.30, 0.15)}


def side_push(env, env_ids, push_xy, asset_cfg=None):
    asset = env.scene["robot"]
    vel_w = asset.data.root_link_vel_w[env_ids].clone()
    vel_w[:, 0] += push_xy[0]
    vel_w[:, 1] += -push_xy[1] * side_t[env_ids]
    asset.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)


def set_conditions(env):
    tw = env.command_manager.get_term("twist")
    foot = env.command_manager.get_term("foot_target")
    hand = env.command_manager.get_term("hand_target")
    tw.vel_command_b.copy_(twist_t)
    tw.vel_command_w.copy_(twist_t)
    for f in ("is_heading_env", "is_standing_env", "is_world_env", "is_forward_env", "is_rotation_env"):
        getattr(tw, f).fill_(False)
    tw.time_left.fill_(float("inf"))
    foot.foot_target_offset_b.copy_(foot_t)
    foot.is_single_support_env.copy_(foot_t.norm(dim=-1).gt(0.0).any(dim=-1))
    foot.lifted_foot_idx.copy_(foot_t.norm(dim=-1).argmax(dim=-1))
    foot.time_left.fill_(float("inf"))
    hand.hand_target_offset_b.copy_(hand_t)
    hand.is_active.copy_(act_t)
    hand.time_left.fill_(float("inf"))


rows = []
t0 = time.time()
for pk in a.pushes.split(","):
    cfg = ev._tracking_cfg(seed=SEEDS[0], steps=STEPS, perturbation=False)
    cfg.scene.num_envs = N
    cfg.auto_reset = True  # a fallen env is masked from then on
    if PUSH[pk] is not None:
        training = make_microban_teleop_v12_env_cfg(play=False)
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
            set_conditions(env)
            obs = ev._patch_initial_command_observation(wrapped.get_observations(), env)
            fell = torch.zeros(N, dtype=torch.bool, device=dev)
            vsum = torch.zeros(N, 3, device=dev)
            cnt = torch.zeros(N, device=dev)
            for step in range(STEPS):
                with torch.inference_mode():
                    act = policy(TensorDict({"actor": obs["actor"]}, batch_size=[N]))
                obs, _r, done, _e = wrapped.step(act)
                nf = done.bool().reshape(-1) | env.termination_manager.get_term("fell_over").bool() | (
                    robot.data.root_link_pos_w[:, 2] < fall_h)
                fell = fell | nf
                if step >= SETTLE:
                    lin = home_levelled_root_lin_vel_b(env, HOME_TRUNK_PITCH_RAD)
                    ang = home_levelled_root_ang_vel_b(env, HOME_TRUNK_PITCH_RAD)
                    v = torch.stack((lin[:, 0], lin[:, 1], ang[:, 2]), dim=-1)
                    m = (~fell).float()
                    vsum += v * m[:, None]
                    cnt += m
            mean = (vsum / cnt.clamp(min=1)[:, None]).tolist()
            for i, (c, t) in enumerate(conds):
                rows.append({"push": pk, "seed": seed, "cmd": c, "targets": t, "twist": COMMANDS[c],
                             "fell": bool(fell[i]), "up_steps": int(cnt[i]), "mean": mean[i]})
            print(pk, seed, "done", round(time.time() - t0, 1), flush=True)
    finally:
        wrapped.close()


def ratio_metrics(cmd, v):
    c = [cmd[i] / SCALE[i] for i in range(3)]
    w = [v[i] / SCALE[i] for i in range(3)]
    n = math.sqrt(sum(x * x for x in c))
    if n == 0:
        return {"angle_deg": None, "speed": None, "perp": math.sqrt(sum(x * x for x in w))}
    u = [x / n for x in c]
    along = sum(w[i] * u[i] for i in range(3))
    perp = math.sqrt(max(0.0, sum(x * x for x in w) - along * along))
    vn = math.sqrt(sum(x * x for x in w))
    angle = math.degrees(math.acos(max(-1.0, min(1.0, along / vn)))) if vn > 1e-9 else 180.0
    return {"angle_deg": angle, "speed": min(max(along, 0.0), n) / n, "perp": perp}


for r in rows:
    r.update(ratio_metrics(r["twist"], r["mean"]))
json.dump({"ckpt": a.ckpt, "iteration": iteration, "seeds": SEEDS, "reps": a.reps, "rows": rows,
           "secs": time.time() - t0}, open(a.out, "w"), indent=1)
print("DONE", round(time.time() - t0, 1))
