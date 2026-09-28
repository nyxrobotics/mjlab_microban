"""Diagnose static HOME behavior under reproducible perturbations.

All candidates run together in the same GPU simulation. The same seeded
21-joint target waveform, initial joint-position error, and body push are
applied to every candidate at both walking (P=125) and A-hold (P=900) gains.
The robot and deployed walking policy are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import mjlab_microban.tasks  # noqa: F401
from mjlab_microban.robot import microban_constants
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.torch import configure_torch_backends
from mjlab_microban.robot.microban_constants import MICROBAN_XML
from mjlab_microban.scripts.home_search_utils import (
    Candidate,
    configure_env,
    contact_root_height_m,
    install_candidate_homes,
    parse_floats,
    parse_ints,
    write_csv,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidates(args: argparse.Namespace) -> list[Candidate]:
    import mujoco

    hips = parse_floats(args.hip_values)
    ankles = parse_floats(args.ankle_values)
    if not hips or not ankles:
        raise ValueError("Provide nonempty --hip-values and --ankle-values")
    if not math.isfinite(args.shoulder_deg):
        raise ValueError("Shoulder pitch must be finite")
    model = mujoco.MjModel.from_xml_path(str(MICROBAN_XML))
    data = mujoco.MjData(model)
    poses = []
    for hip in hips:
        for ankle in ankles:
            if not -20.0 <= hip <= 5.0 or not -10.0 <= ankle <= 10.0:
                raise ValueError("Search bound: hip [-20,5], ankle [-10,10] deg")
            raw = Candidate(f"h{hip:+g}_a{ankle:+g}_s{args.shoulder_deg:+g}", hip, ankle, args.shoulder_deg)
            poses.append(
                Candidate(raw.name, hip, ankle, args.shoulder_deg,
                          contact_root_height_m(model, data, raw))
            )
    return poses


def _configure_static_env(
    num_envs: int,
    duration_s: float,
    kp_fw: int,
    voltage_v: float,
    drop_gain_v_per_nm: float,
    delay_steps: int,
):
    cfg = configure_env(num_envs, duration_s)
    # BAM samples voltage and voltage drop once at environment construction,
    # before trial-level env.reset(seed=...) runs. Make every candidate see
    # the same nominal actuator instead of confounding HOME with motor draws.
    cfg.seed = 0
    cfg.auto_reset = False
    cfg.terminations = {}
    cfg.rewards = {}
    actuator = cfg.scene.entities["robot"].articulation.actuators[0]
    actuator.kp_fw = kp_fw
    actuator.vin_range = (voltage_v, voltage_v)
    actuator.vin_drop_gain_range = (drop_gain_v_per_nm, drop_gain_v_per_nm)
    actuator.delay_min_lag = delay_steps
    actuator.delay_max_lag = delay_steps
    return cfg


def _run_trial(
    env: ManagerBasedRlEnv,
    poses: list[Candidate],
    *,
    kp_fw: int,
    level: float,
    seed: int,
    steps: int,
    push_tick: int,
    target_wave_rad: float,
    initial_perturb_rad: float,
    push_linear_m_s: float,
    push_angular_rad_s: float,
) -> list[dict]:
    env.reset(seed=seed)
    robot = env.scene["robot"]
    defaults = robot.data.default_joint_pos
    n, joint_count = defaults.shape
    rng = np.random.default_rng(seed)
    initial = torch.tensor(
        rng.uniform(-initial_perturb_rad, initial_perturb_rad, joint_count),
        device=env.device, dtype=defaults.dtype,
    )
    joint_position = defaults + initial
    joint_position = torch.maximum(
        torch.minimum(joint_position, robot.data.joint_pos_limits[..., 1]),
        robot.data.joint_pos_limits[..., 0],
    )
    initial_clip_counts = (
        (joint_position != defaults + initial).sum(dim=1).detach().cpu().tolist()
    )
    robot.write_joint_position_to_sim(joint_position)
    robot.set_joint_position_target(defaults)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.sim.sense()

    # Each joint receives a distinct low-frequency target signal; the same
    # realization is broadcast to all candidate homes and reused for both KP.
    frequencies = torch.tensor(rng.uniform(0.45, 1.8, joint_count), device=env.device)
    phases = torch.tensor(rng.uniform(0.0, 2.0 * math.pi, joint_count), device=env.device)
    amplitudes = torch.tensor(
        rng.uniform(0.5, 1.0, joint_count), device=env.device
    ) * target_wave_rad
    push_direction = rng.normal(size=4)
    push_direction /= max(1e-12, float(np.linalg.norm(push_direction)))
    push_delta = torch.tensor(
        [
            push_direction[0] * push_linear_m_s * level,
            push_direction[1] * push_linear_m_s * level,
            0.0,
            push_direction[2] * push_angular_rad_s * level,
            push_direction[3] * push_angular_rad_s * level,
            0.0,
        ],
        device=env.device,
        dtype=defaults.dtype,
    )
    action = env.action_manager.get_term("joint_pos")
    actor_joint_ids = action._target_ids
    raw_actions = torch.zeros((n, action.action_dim), device=env.device)

    fallen = torch.zeros(n, dtype=torch.bool, device=env.device)
    survival_steps = torch.full((n,), steps, dtype=torch.int64, device=env.device)
    tilt_sum = torch.zeros(n, dtype=torch.float64, device=env.device)
    tilt_max = torch.zeros(n, dtype=torch.float64, device=env.device)
    joint_error_sum = torch.zeros(n, dtype=torch.float64, device=env.device)
    torque_sum = torch.zeros(n, dtype=torch.float64, device=env.device)
    samples = torch.zeros(n, dtype=torch.float64, device=env.device)
    recover_streak = torch.zeros(n, dtype=torch.int64, device=env.device)
    recover_tick = torch.full((n,), steps, dtype=torch.int64, device=env.device)
    recovered = torch.zeros(n, dtype=torch.bool, device=env.device)
    min_root_z = torch.full((n,), float("inf"), device=env.device)
    push_applied = False
    recovery_streak_needed = max(1, math.ceil(0.2 / env.step_dt))

    for tick in range(steps):
        if tick == push_tick and level > 0:
            velocity = robot.data.root_link_vel_w.clone() + push_delta
            robot.write_root_link_velocity_to_sim(velocity)
            push_applied = True
        time_s = tick * env.step_dt
        ramp = min(1.0, time_s / 0.5)
        wave = ramp * amplitudes * torch.sin(2.0 * math.pi * frequencies * time_s + phases)
        target = defaults + wave.unsqueeze(0)
        target = torch.maximum(
            torch.minimum(target, robot.data.soft_joint_pos_limits[..., 1]),
            robot.data.soft_joint_pos_limits[..., 0],
        )
        robot.set_joint_position_target(target)
        raw_actions.copy_(target[:, actor_joint_ids] - action.offset)
        # No termination term or auto-reset is active: each candidate's first
        # fall is recorded and its remaining trajectory is ignored.
        env.step(raw_actions)
        grav_z = robot.data.projected_gravity_b[:, 2]
        tilt = torch.rad2deg(torch.acos(torch.clamp(-grav_z, -1.0, 1.0)))
        root_z = robot.data.root_link_pos_w[:, 2]
        min_root_z = torch.minimum(min_root_z, root_z)
        finite_state = torch.isfinite(tilt) & torch.isfinite(root_z)
        now_fallen = (grav_z > -0.5) | (root_z < 0.10) | ~finite_state
        fresh_fall = ~fallen & now_fallen
        survival_steps[fresh_fall] = tick + 1
        fallen |= fresh_fall
        alive = ~fallen
        alive_float = alive.to(dtype=torch.float64)
        safe_tilt = torch.nan_to_num(tilt, nan=180.0, posinf=180.0, neginf=180.0)
        tilt_sum += safe_tilt.to(dtype=torch.float64) * alive_float
        tilt_max = torch.maximum(tilt_max, torch.where(alive, safe_tilt.double(), 0.0))
        joint_error = torch.rad2deg(
            torch.linalg.vector_norm(robot.data.joint_pos - target, dim=1) / math.sqrt(joint_count)
        )
        safe_joint_error = torch.nan_to_num(joint_error, nan=180.0, posinf=180.0, neginf=180.0)
        safe_torque = torch.nan_to_num(
            robot.data.qfrc_actuator.abs().mean(dim=1), nan=0.0, posinf=0.0, neginf=0.0
        )
        joint_error_sum += safe_joint_error.double() * alive_float
        torque_sum += safe_torque.double() * alive_float
        samples += alive_float
        if push_applied:
            near_upright = (tilt < 10.0) & (joint_error < 3.0) & alive
            recover_streak = torch.where(near_upright, recover_streak + 1, 0)
            newly_recovered = ~recovered & (recover_streak >= recovery_streak_needed)
            recover_tick[newly_recovered] = tick + 1
            recovered |= newly_recovered
        if bool(fallen.all().item()):
            break

    def floats(tensor: torch.Tensor) -> list[float]:
        return [float(value) for value in tensor.detach().cpu().tolist()]

    survival = floats(survival_steps)
    tilt_total = floats(tilt_sum)
    tilt_peak = floats(tilt_max)
    error_total = floats(joint_error_sum)
    torque_total = floats(torque_sum)
    counts = floats(samples)
    recovery = floats(recover_tick)
    roots = floats(min_root_z)
    falls = fallen.detach().cpu().tolist()
    recovers = recovered.detach().cpu().tolist()
    rows = []
    for index, pose in enumerate(poses):
        count = counts[index]
        rows.append({
            "candidate": pose.name,
            "kp_fw": kp_fw,
            "push_level": level,
            "seed": seed,
            "fell": bool(falls[index]),
            "survival_s": survival[index] * env.step_dt,
            "recovered_after_push": bool(recovers[index]) if push_applied else None,
            "recovery_s": (
                max(0.0, (recovery[index] - push_tick - recovery_streak_needed) * env.step_dt)
                if recovers[index] and push_applied else None
            ),
            "mean_tilt_deg": tilt_total[index] / count if count else None,
            "max_tilt_deg": tilt_peak[index],
            "mean_joint_target_error_deg": error_total[index] / count if count else None,
            "mean_abs_joint_torque_nm": torque_total[index] / count if count else None,
            "min_root_z_m": roots[index],
            "initial_joint_clip_count": int(initial_clip_counts[index]),
        })
    return rows


def _summary(poses: list[Candidate], trials: list[dict], duration_s: float, push_tick_s: float) -> list[dict]:
    def mean(values: list[float | None]) -> float | None:
        found = [float(value) for value in values if value is not None and math.isfinite(value)]
        return sum(found) / len(found) if found else None

    rows = []
    for pose in poses:
        subset = [trial for trial in trials if trial["candidate"] == pose.name]
        pushed = [trial for trial in subset if trial["push_level"] > 0]
        # A missed recovery is charged the remaining horizon. This is a
        # censored time-to-recover metric, not an invented observed recovery.
        recovery_penalties = [
            trial["recovery_s"] if trial["recovery_s"] is not None else duration_s - push_tick_s
            for trial in pushed
        ]
        rows.append({
            **asdict(pose),
            "trials": len(subset),
            "falls": sum(trial["fell"] for trial in subset),
            "fall_rate": mean([float(trial["fell"]) for trial in subset]),
            "mean_survival_s": mean([trial["survival_s"] for trial in subset]),
            "post_push_recovery_rate": mean([
                float(trial["recovered_after_push"]) for trial in pushed
            ]),
            "mean_censored_recovery_s": mean(recovery_penalties),
            "mean_tilt_deg": mean([trial["mean_tilt_deg"] for trial in subset]),
            "max_tilt_deg": max(trial["max_tilt_deg"] for trial in subset),
            "mean_joint_target_error_deg": mean([
                trial["mean_joint_target_error_deg"] for trial in subset
            ]),
            "mean_abs_joint_torque_nm": mean([
                trial["mean_abs_joint_torque_nm"] for trial in subset
            ]),
        })
    rows.sort(key=lambda row: (
        row["falls"],
        -row["mean_survival_s"],
        row["mean_censored_recovery_s"] if row["mean_censored_recovery_s"] is not None else float("inf"),
        row["mean_tilt_deg"] if row["mean_tilt_deg"] is not None else float("inf"),
    ))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hip-values", default="-12,-10,-8")
    parser.add_argument("--ankle-values", default="-2,0,2")
    parser.add_argument("--shoulder-deg", type=float, default=10.0)
    parser.add_argument("--kp-values", default="125,900")
    parser.add_argument("--push-levels", default="0,1,2")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--push-at-s", type=float, default=1.0)
    parser.add_argument("--target-wave-deg", type=float, default=1.0)
    parser.add_argument("--initial-joint-error-deg", type=float, default=0.5)
    parser.add_argument("--push-linear-m-s", type=float, default=0.08)
    parser.add_argument("--push-angular-rad-s", type=float, default=0.4)
    parser.add_argument("--voltage-v", type=float, default=10.8)
    parser.add_argument("--drop-gain-v-per-nm", type=float, default=0.1)
    parser.add_argument("--delay-steps", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-prefix", type=Path, default=Path("artifacts/home_search/static_robust"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.push_at_s < args.duration_s:
        raise ValueError("Require 0 < push-at-s < duration-s")
    if min(args.target_wave_deg, args.initial_joint_error_deg, args.push_linear_m_s,
           args.push_angular_rad_s) < 0:
        raise ValueError("Perturbation amplitudes must be nonnegative")
    if not 9.0 <= args.voltage_v <= 12.6:
        raise ValueError("Voltage must be within the configured 3S range [9.0,12.6] V")
    if not 0.0 <= args.drop_gain_v_per_nm <= 0.2:
        raise ValueError("Drop gain must be within the configured range [0,0.2] V/Nm")
    if args.delay_steps < 0:
        raise ValueError("Delay steps must be nonnegative")
    kp_values = parse_ints(args.kp_values)
    if any(kp <= 0 for kp in kp_values):
        raise ValueError("KP values must be positive")
    levels = parse_floats(args.push_levels)
    if any(level < 0 for level in levels):
        raise ValueError("Push levels must be nonnegative")
    seeds = parse_ints(args.seeds)
    poses = _candidates(args)
    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    trials = []
    control_period_s = None
    for kp_fw in kp_values:
        cfg = _configure_static_env(
            len(poses), args.duration_s, kp_fw,
            args.voltage_v, args.drop_gain_v_per_nm, args.delay_steps,
        )
        env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
        try:
            install_candidate_homes(env, poses)
            control_period_s = env.step_dt
            steps = math.ceil(args.duration_s / env.step_dt)
            push_tick = math.floor(args.push_at_s / env.step_dt)
            for level in levels:
                for seed in seeds:
                    print(f"[INFO] KP={kp_fw} push={level:g} seed={seed} homes={len(poses)}", flush=True)
                    trials.extend(_run_trial(
                        env, poses, kp_fw=kp_fw, level=level, seed=seed,
                        steps=steps, push_tick=push_tick,
                        target_wave_rad=math.radians(args.target_wave_deg),
                        initial_perturb_rad=math.radians(args.initial_joint_error_deg),
                        push_linear_m_s=args.push_linear_m_s,
                        push_angular_rad_s=args.push_angular_rad_s,
                    ))
        finally:
            env.close()
    summary = _summary(poses, trials, args.duration_s, args.push_at_s)
    output = args.output_prefix
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path = output.with_suffix(".json")
    trial_csv = output.with_name(output.name + "_trials.csv")
    summary_csv = output.with_name(output.name + "_summary.csv")
    report = {
        "metadata": {
            "method": "paired perturbation search with static 21-joint HOME targets",
            "robot_xml": str(MICROBAN_XML),
            "robot_xml_sha256": _sha256(MICROBAN_XML),
            "robot_constants_sha256": _sha256(Path(microban_constants.__file__)),
            "device": args.device,
            "control_period_s": control_period_s,
            "kp_fw": kp_values,
            "push_levels": levels,
            "seeds": seeds,
            "duration_s": args.duration_s,
            "push_at_s": args.push_at_s,
            "target_wave_deg": args.target_wave_deg,
            "initial_joint_error_deg": args.initial_joint_error_deg,
            "push_linear_m_s": args.push_linear_m_s,
            "push_angular_rad_s": args.push_angular_rad_s,
            "actuator_voltage_v": args.voltage_v,
            "actuator_drop_gain_v_per_nm": args.drop_gain_v_per_nm,
            "actuator_delay_steps": args.delay_steps,
            "environment_init_seed": 0,
            "fall_definition": "root z < 0.10 m or projected gravity z > -0.5",
            "rank_order": "fewest falls, longest survival, shortest censored recovery, least mean tilt",
            "policy": "none; static HOME holding controller",
        },
        "candidates": [asdict(pose) for pose in poses],
        "trials": trials,
        "summary": summary,
    }
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(trial_csv, trials)
    write_csv(summary_csv, summary)
    print(f"[INFO] {json_path}\n[INFO] {trial_csv}\n[INFO] {summary_csv}")
    print(json.dumps(summary[:5], indent=2))


if __name__ == "__main__":
    main()
