"""Why did the B3 walker stop turning right at update 4000?  (diagnostic only)

For each checkpoint: yaw-only commands right (0, 0, -0.5) and left (0, 0, +0.5),
5 envs each, seeds 101-103, 300 steps in the walking env's play config.
Measured (window 50..299): yaw rate, lateral/forward drift, vertical bob, roll/pitch
rates, touchdowns per second (stepping or standing), and the B3 reward breakdown on
the 0.5 s filtered motion (speed, deviation, backward, total) next to the reward of
standing still and of turning exactly as commanded; the same for B4.

usage: yaw_diag.py OUT.json CKPT [CKPT ...]   (--device cpu|cuda:0)
"""
import argparse
import hashlib
import json
import math

import torch
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends

from mjlab_microban.robot.microban_constants import HOME_TRUNK_PITCH_RAD
from mjlab_microban.scripts.teleop_v12_bootstrap_gate import _legacy_model
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import inspect_legacy_velocity_checkpoint
from mjlab_microban.tasks.microban_twist_ratio_mdp import home_levelled_velocities, twist_ratio, twist_ratio_value
from mjlab_microban.tasks.microban_velocity_env_cfg import make_microban_velocity_env_cfg

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("ckpts", nargs="+")
p.add_argument("--device", default="cpu")
p.add_argument("--seeds", default="101,102,103")
a = p.parse_args()
dev = a.device
STEPS, SETTLE, REPS = 300, 50, 5
TAU = 0.5
configure_torch_backends(allow_tf32=False, deterministic=True)
CMDS = {"YR": (0.0, 0.0, -0.5), "YL": (0.0, 0.0, 0.5)}
conds = [c for c in CMDS for _ in range(REPS)]
N = len(conds)
twist_t = torch.tensor([CMDS[c] for c in conds], device=dev)


def b3_value(parts):
    error = parts.deviation + parts.backward * parts.command_norm.clamp(min=0.2)  # B3 counted |p-|, not /n
    return 0.5 * (1.0 + parts.speed) * torch.exp(-error)


def set_command(env):
    tw = env.command_manager.get_term("twist")
    tw.vel_command_b.copy_(twist_t)
    for f in ("is_heading_env", "is_standing_env", "is_rotation_env"):
        if hasattr(tw, f):
            getattr(tw, f).fill_(False)
    tw.time_left.fill_(float("inf"))


results = {}
cfg = make_microban_velocity_env_cfg(play=True)
cfg.scene.num_envs = N
cfg.auto_reset = True
cfg.episode_length_s = (STEPS + 2) * cfg.decimation * cfg.sim.mujoco.timestep
cfg.commands["twist"].resampling_time_range = (1.0e6, 1.0e6)
cfg.commands["twist"].debug_vis = False
cfg.events.pop("push_robot", None)
env = ManagerBasedRlEnv(cfg=cfg, device=dev)
wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
robot = env.scene["robot"]
contact = env.scene["feet_ground_contact"]
gain = 1.0 - math.exp(-env.step_dt / TAU)
for ck in a.ckpts:
    sha = hashlib.sha256(open(ck, "rb").read()).hexdigest()
    identity, state = inspect_legacy_velocity_checkpoint(ck, sha)
    policy = _legacy_model().to(dev)
    policy.load_state_dict(state, strict=True)
    policy.eval()
    acc = {c: {"twist": [], "unc": [], "touchdowns": 0.0, "b3": [], "b3_parts": [], "b4": [], "n": 0, "fell": 0} for c in CMDS}
    for seed in [int(s) for s in a.seeds.split(",")]:
        env.reset(seed=seed)
        set_command(env)
        obs = TensorDict(env.observation_manager.compute(update_history=True), batch_size=[N])
        fell = torch.zeros(N, dtype=torch.bool, device=dev)
        f_cmd = f_tw = f_unc = None
        prev_contact = None
        tdowns = torch.zeros(N, device=dev)
        rec = {c: [] for c in ("twist", "unc", "b3", "spd", "dev", "bwd", "b4")}
        for step in range(STEPS):
            set_command(env)
            with torch.inference_mode():
                act = policy(TensorDict({"actor": obs["actor"]}, batch_size=[N]))
            obs, _r, done, _e = wrapped.step(act)
            fell |= done.bool().reshape(-1)
            lin, ang = home_levelled_velocities(env, HOME_TRUNK_PITCH_RAD)
            tw = torch.stack((lin[:, 0], lin[:, 1], ang[:, 2]), -1)
            unc = torch.stack((lin[:, 2], ang[:, 0], ang[:, 1]), -1)
            if f_tw is None:
                f_cmd, f_tw, f_unc = twist_t.clone(), tw.clone(), unc.clone()
            else:
                f_cmd = f_cmd + gain * (twist_t - f_cmd)
                f_tw = f_tw + gain * (tw - f_tw)
                f_unc = f_unc + gain * (unc - f_unc)
            found = contact.data.found.reshape(N, -1) > 0
            if prev_contact is not None and step >= SETTLE:
                tdowns += (found & ~prev_contact).sum(-1).float()
            prev_contact = found
            if step >= SETTLE:
                parts = twist_ratio(f_cmd, f_tw, uncommanded=f_unc)
                rec["twist"].append(tw); rec["unc"].append(unc)
                rec["b3"].append(b3_value(parts)); rec["spd"].append(parts.speed)
                rec["dev"].append(parts.deviation); rec["bwd"].append(parts.backward)
                rec["b4"].append(twist_ratio_value(parts))
        stack = {k: torch.stack(v) for k, v in rec.items()}
        for i, c in enumerate(conds):
            a_ = acc[c]
            a_["n"] += 1
            a_["fell"] += int(fell[i])
            a_["twist"].append(stack["twist"][:, i].mean(0).tolist())
            a_["unc"].append(stack["unc"][:, i].abs().mean(0).tolist())
            a_["touchdowns"] += float(tdowns[i]) / ((STEPS - SETTLE) * env.step_dt)
            a_["b3"].append(float(stack["b3"][:, i].mean()))
            a_["b3_parts"].append([float(stack["spd"][:, i].mean()), float(stack["dev"][:, i].mean()), float(stack["bwd"][:, i].mean())])
            a_["b4"].append(float(stack["b4"][:, i].mean()))
    out = {}
    for c, a_ in acc.items():
        mean = lambda xs: [sum(x[j] for x in xs) / len(xs) for j in range(len(xs[0]))]
        tw = mean(a_["twist"]); unc = mean(a_["unc"]); parts_m = mean(a_["b3_parts"])
        cmd = torch.tensor([CMDS[c]])
        # Hypotheticals on the filtered-equivalent mean motion.
        drift = torch.tensor([[tw[0], tw[1], 0.0]])
        still = twist_ratio(cmd, torch.zeros(1, 3), uncommanded=torch.zeros(1, 3))
        exact = twist_ratio(cmd, cmd.clone(), uncommanded=torch.zeros(1, 3))
        exact_drift = twist_ratio(cmd, cmd + drift, uncommanded=torch.tensor([unc]))
        out[c] = {
            "rollouts": a_["n"], "fell": a_["fell"],
            "mean_twist_vx_vy_wz": tw, "mean_abs_vz_wx_wy": unc,
            "touchdowns_per_s": a_["touchdowns"] / a_["n"],
            "B3_reward_mean": sum(a_["b3"]) / len(a_["b3"]),
            "B3_parts_speed_deviation_backward": parts_m,
            "B4_reward_mean": sum(a_["b4"]) / len(a_["b4"]),
            "B3_if_standing_still": float(b3_value(still)[0]), "B3_if_exact_turn": float(b3_value(exact)[0]),
            "B3_if_turn_with_measured_drift_and_bob": float(b3_value(exact_drift)[0]),
            "B4_if_standing_still": float(twist_ratio_value(still)[0]), "B4_if_exact_turn": float(twist_ratio_value(exact)[0]),
            "B4_if_turn_with_measured_drift_and_bob": float(twist_ratio_value(exact_drift)[0]),
        }
    results[ck] = {"iteration": identity.iteration, **out}
    print(json.dumps({"iteration": identity.iteration, **out}, indent=1), flush=True)
wrapped.close()
json.dump(results, open(a.out, "w"), indent=1)
