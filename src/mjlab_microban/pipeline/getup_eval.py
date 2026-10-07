"""Evaluate a get-up checkpoint (stand-up / push / standing posture) at HOME.

usage:
  uv run --locked python -m mjlab_microban.pipeline.getup_eval stand TASK CKPT \
      [--envs 64] [--seed 11] [--steps 999] [--imu-delay 3] [--noise] [--push x:0.3]
  uv run --locked python -m mjlab_microban.pipeline.getup_eval posture TASK CKPT \
      [--envs 64] [--seed 11] [--steps 999]

The get-up evaluations of the 2026-10 retraining (eval_v4.py /
posture_v4.py): human-readable lines, then one ``RESULT {json}`` line that
the pipeline parses.  TASK's play config (full IMU latency, no schedule) is
the scene.
Tilt is measured from HOME's trunk orientation (HOME_TRUNK_PITCH_RAD), so the
numbers mean the same at a pitched-trunk HOME.

stand: fallen and near-HOME starts, one 20 s episode; standing = the head
above 0.9 x HEAD_STANDING_HEIGHT on every step of the final second.
posture: fallen starts only; joint means of the envs standing through the
final 5 s, compared with HOME.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

from mjlab_microban.robot.microban_constants import HOME_FRAME, HOME_TRUNK_PITCH_RAD
from mjlab_microban.tasks.mdp import _head_height
from mjlab_microban.tasks.microban_getup_env_cfg import GETUP_ACTION_CLIP_RAD, HEAD_STANDING_HEIGHT

DT = 0.02
THR = 0.9 * HEAD_STANDING_HEIGHT


def _setup(task: str, ckpt: str, envs: int, seed: int, *, posture: bool, delay: int, noise: bool):
    torch.manual_seed(seed)
    env_cfg = load_env_cfg(task, play=True)
    env_cfg.scene.num_envs = envs
    env_cfg.seed = seed
    if posture:
        env_cfg.events["reset_near_home"].params["rel_near_home_envs"] = 0.0
    else:
        if noise:
            env_cfg.observations["actor"].enable_corruption = True
        for name in ("base_ang_vel", "projected_gravity"):
            term = env_cfg.observations["actor"].terms[name]
            term.delay_min_lag = 0
            term.delay_max_lag = delay
            term.delay_update_period = 64
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    wrapped = RslRlVecEnvWrapper(env)
    runner = load_runner_cls(task)(wrapped, asdict(load_rl_cfg(task)), device="cuda:0")
    runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    return env, wrapped, runner.get_inference_policy(device="cuda:0")


def _tilt_deg(g: torch.Tensor, g_home: torch.Tensor) -> torch.Tensor:
    return torch.rad2deg(torch.acos(torch.clamp((g * g_home).sum(-1), -1, 1)))


def stand(args) -> dict:
    n, steps = args.envs, args.steps
    env, wrapped, policy = _setup(args.task, args.checkpoint, n, args.seed, posture=False,
                                  delay=args.imu_delay, noise=args.noise)
    g_home = torch.tensor([math.sin(HOME_TRUNK_PITCH_RAD), 0.0, -math.cos(HOME_TRUNK_PITCH_RAD)],
                          device="cuda:0")
    robot = env.scene["robot"]
    ids = env.action_manager.get_term("joint_pos").target_ids
    names = [robot.joint_names[i] for i in ids.tolist()]
    home = torch.tensor([HOME_FRAME.joint_pos[name] for name in names], device="cuda:0")
    head_cfg = SceneEntityCfg("robot", body_names=("head",))
    hands = env.scene["hands_ground_contact"]
    feet = env.scene["feet_ground_contact"]
    action_term = env.action_manager.get_term("joint_pos")
    h, hand_c, feet_c, tilt, dev, raw_abs, on_clip, past_1p57, jvel = (torch.zeros(steps, n) for _ in range(9))
    push_step = min(400, steps // 2)
    pre_push_standing = None
    obs = wrapped.get_observations()
    for t in range(steps):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = wrapped.step(actions)
        if args.push and t == push_step:
            axis, value = args.push.split(":")
            pre_push_standing = (_head_height(env, head_cfg) > THR).cpu()
            sign = torch.where(torch.rand(n, device="cuda:0") > 0.5, 1.0, -1.0)
            vel = robot.data.root_link_vel_w.clone()
            vel[:, 0 if axis == "x" else 1] += sign * float(value)
            robot.write_root_link_velocity_to_sim(vel)
        h[t] = _head_height(env, head_cfg).cpu()
        hand_c[t] = (hands.data.found > 0).any(dim=-1).float().cpu()
        feet_c[t] = (feet.data.found > 0).float().sum(dim=-1).cpu()
        tilt[t] = _tilt_deg(robot.data.projected_gravity_b, g_home).cpu()
        q = robot.data.joint_pos[:, ids]
        dev[t] = torch.sqrt(torch.mean((q - home) ** 2, dim=-1)).rad2deg().cpu()
        raw_abs[t] = actions.abs().mean(dim=-1).cpu()
        target = action_term._processed_actions.abs()
        # On the clip: the target saturated at the servo range (+-pi), the
        # bound the reward's clip barrier charges beyond and the robot clips at.
        on_clip[t] = (target >= GETUP_ACTION_CLIP_RAD - 1.0e-4).float().mean(dim=-1).cpu()
        # |target| >= 1.57 rad (the old clip, before 2026-10-03): recorded to
        # compare with the 2026-10 history; the reward allows it (torque authority).
        past_1p57[t] = (target >= 1.5699).float().mean(dim=-1).cpu()
        jvel[t] = robot.data.joint_vel[:, ids].abs().mean(dim=-1).cpu()
    env.close()

    near_home = dev[0] < 8.0
    fallen = ~near_home
    win = min(50, steps // 4)
    final = h[-win:]
    standing_end = (final > THR).all(dim=0)
    stand_mask = h > THR

    def masked_mean(x):
        return float((x * stand_mask).sum() / stand_mask.sum().clamp(min=1))

    result = {
        "mode": "stand", "task": args.task, "checkpoint": args.checkpoint, "envs": n,
        "seed": args.seed, "steps": steps, "imu_delay": args.imu_delay, "noise": args.noise,
        "push": args.push,
        "fallen_starts": int(fallen.sum()),
        "fallen_standing_end": int((standing_end & fallen).sum()),
        "all_standing_end": int(standing_end.sum()),
        "fallen_standing_fraction": float((standing_end & fallen).sum()) / max(1, int(fallen.sum())),
        "final_head_height_m": float(final.mean()),
        "final_tilt_deg": float(tilt[-win:].mean()),
        "final_feet_contacts": float(feet_c[-win:].mean()),
        "final_hands_on_ground": float(hand_c[-win:].mean()),
        "standing_rms_dev_from_home_deg": masked_mean(dev),
        "standing_joint_abs_vel_rad_s": masked_mean(jvel),
        "standing_targets_on_clip": masked_mean(on_clip),
        "standing_targets_beyond_1p57": masked_mean(past_1p57),
        "standing_raw_action_abs": masked_mean(raw_abs),
    }
    if pre_push_standing is not None:
        after = h[push_step + 1:min(steps, push_step + 151)]
        fell = (after.min(dim=0).values < 0.20) & pre_push_standing if len(after) else pre_push_standing & False
        result["push_standing_before"] = int(pre_push_standing.sum())
        result["push_fell_within_3s"] = int(fell.sum())
        result["push_standing_end"] = int((pre_push_standing & standing_end).sum())
    print(f"TASK={args.task} ckpt={args.checkpoint} envs={n} seed={args.seed} delay=0-{args.imu_delay} "
          f"noise={args.noise} push={args.push}")
    print(f"  STANDING at end: all {result['all_standing_end']}/{n}, fallen-start "
          f"{result['fallen_standing_end']}/{result['fallen_starts']}")
    print(f"  final: head {result['final_head_height_m']:.3f} m, tilt {result['final_tilt_deg']:.1f} deg; "
          f"while standing: |vel| {result['standing_joint_abs_vel_rad_s']:.2f} rad/s, on the +-pi clip "
          f"{result['standing_targets_on_clip']:.2f}, |target|>1.57 {result['standing_targets_beyond_1p57']:.2f}, "
          f"|raw| {result['standing_raw_action_abs']:.1f}")
    if "push_fell_within_3s" in result:
        print(f"  PUSH {args.push}: standing before {result['push_standing_before']}, fell within 3 s "
              f"{result['push_fell_within_3s']}, standing at end {result['push_standing_end']}")
    return result


def posture(args) -> dict:
    n, steps = args.envs, args.steps
    env, wrapped, policy = _setup(args.task, args.checkpoint, n, args.seed, posture=True,
                                  delay=0, noise=False)
    g_home = torch.tensor([math.sin(HOME_TRUNK_PITCH_RAD), 0.0, -math.cos(HOME_TRUNK_PITCH_RAD)],
                          device="cuda:0")
    robot = env.scene["robot"]
    head_cfg = SceneEntityCfg("robot", body_names=("head",))
    ids = env.action_manager.get_term("joint_pos").target_ids
    names = [robot.joint_names[i] for i in ids.tolist()]
    home = torch.tensor([HOME_FRAME.joint_pos[name] for name in names], device="cuda:0")
    feet_ids = [robot.body_names.index(b) for b in ("foot", "foot_2")]
    hs, qs, fds, tilts = [], [], [], []
    obs = wrapped.get_observations()
    for _ in range(steps):
        with torch.inference_mode():
            obs, _, _, _ = wrapped.step(policy(obs))
        hs.append(_head_height(env, head_cfg))
        qs.append(robot.data.joint_pos[:, ids].clone())
        p = robot.data.body_link_pos_w[:, feet_ids, :]
        d = quat_apply_inverse(yaw_quat(robot.data.root_link_quat_w), p[:, 0, :] - p[:, 1, :])
        fds.append(d[:, :2].abs())
        tilts.append(_tilt_deg(robot.data.projected_gravity_b, g_home))
    H, Q, FD, TILT = (torch.stack(x) for x in (hs, qs, fds, tilts))
    win = slice(steps - min(250, steps // 2), steps)
    ok = (H[win] > THR).all(0)
    count = int(ok.sum())
    result = {"mode": "posture", "task": args.task, "checkpoint": args.checkpoint, "envs": n,
              "seed": args.seed, "steps": steps, "standing_final_window": count,
              "standing_fraction": count / n}
    if count:
        q = Q[win][:, ok].mean((0, 1))
        fd = FD[win][:, ok].mean((0, 1))
        result.update(
            head_height_m=float(H[win][:, ok].mean()), tilt_deg=float(TILT[win][:, ok].mean()),
            feet_lateral_m=float(fd[1]), feet_fore_aft_m=float(fd[0]),
            joint_mean_minus_home_deg={nm: float(v) for nm, v in zip(names, (q - home).rad2deg().tolist())},
        )
    env.close()
    print(f"{args.checkpoint}: standing through the final window: {count}/{n}")
    if count:
        print(f"  head {result['head_height_m']:.3f} m (HOME {HEAD_STANDING_HEIGHT}), tilt "
              f"{result['tilt_deg']:.1f} deg, feet lateral {result['feet_lateral_m'] * 100:.1f} cm, "
              f"fore-aft {result['feet_fore_aft_m'] * 100:.1f} cm")
        worst = max(result["joint_mean_minus_home_deg"].items(), key=lambda kv: abs(kv[1]))
        print(f"  largest mean joint offset from HOME: {worst[0]} {worst[1]:+.1f} deg")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("stand", "posture"))
    parser.add_argument("task")
    parser.add_argument("checkpoint")
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=999)
    parser.add_argument("--imu-delay", type=int, default=3)
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--push", default=None, help="axis:m/s velocity kick at 8 s, e.g. x:0.3")
    args = parser.parse_args()
    result = stand(args) if args.mode == "stand" else posture(args)
    print("RESULT " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
