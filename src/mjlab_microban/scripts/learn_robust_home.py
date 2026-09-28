"""Learn a symmetric HOME from all-joint disturbance rollouts in one CEM run.

The continuous parameters are hip, ankle, and shoulder pitch. All candidates
share the same actuator model, position-hold controller, and paired disturbance
seeds. This learns a HOME for the *simulated standing task*, not a walking
policy or a deployable physical robot pose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.torch import configure_torch_backends
from mjlab_microban.robot import microban_constants
from mjlab_microban.robot.microban_constants import MICROBAN_XML
from mjlab_microban.scripts.home_search_utils import (
    Candidate,
    contact_root_height_m,
    install_candidate_homes,
)
from mjlab_microban.scripts.optimize_static_home import _configure_static_env, _run_trial


PARAMETERS = ("hip_pitch_deg", "ankle_pitch_deg", "shoulder_pitch_deg")
LOWER = np.array([-20.0, -10.0, -5.0], dtype=np.float64)
UPPER = np.array([5.0, 10.0, 20.0], dtype=np.float64)
OLD_HOME = np.array([-10.0, 0.0, 10.0], dtype=np.float64)
CENTERED_HOME = np.array([1.198384259489, -1.198384259489, 0.0], dtype=np.float64)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pose(name: str, angles: np.ndarray, model: mujoco.MjModel, data: mujoco.MjData) -> Candidate:
    raw = Candidate(name, *[float(angle) for angle in angles])
    return Candidate(raw.name, raw.hip_deg, raw.ankle_deg, raw.shoulder_deg,
                     contact_root_height_m(model, data, raw))


def _full_home_joint_deg(pose: dict) -> dict[str, float]:
    """Expand the selected symmetric parameters into all 21 HOME joint angles."""
    joints = {
        name: math.degrees(angle)
        for name, angle in microban_constants.HOME_FRAME.joint_pos.items()
    }
    for side in ("left", "right"):
        joints[f"{side}_hip_pitch"] = float(pose["hip_deg"])
        joints[f"{side}_ankle_pitch"] = float(pose["ankle_deg"])
        joints[f"{side}_shoulder_pitch"] = float(pose["shoulder_deg"])
    return joints


def _score(trials: list[dict], duration_s: float) -> dict:
    """Prioritize worst-case survival, then uprightness; no task-specific target pose."""
    def tilt(row: dict) -> float:
        value = row["mean_tilt_deg"]
        return float(value) if value is not None and math.isfinite(value) else 180.0

    per_trial = [
        float(row["survival_s"]) / duration_s
        + (0.1 if not row["fell"] else 0.0)
        - 0.003 * tilt(row)
        for row in trials
    ]
    return {
        "score": float(0.5 * np.mean(per_trial) + 0.5 * min(per_trial)),
        "fall_rate": float(np.mean([row["fell"] for row in trials])),
        "worst_survival_s": float(min(row["survival_s"] for row in trials)),
        "mean_survival_s": float(np.mean([row["survival_s"] for row in trials])),
        "mean_tilt_deg": float(np.mean([tilt(row) for row in trials])),
    }


def _evaluate(
    env: ManagerBasedRlEnv,
    poses: list[Candidate],
    *,
    kp_fw: int,
    seeds: list[int],
    push_levels: list[float],
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict]]:
    robot = env.scene["robot"]
    if robot.data.default_joint_pos.shape[1] != 21:
        raise RuntimeError("This HOME experiment requires exactly 21 actuated joints")
    install_candidate_homes(env, poses)
    trials = []
    for seed in seeds:
        for level in push_levels:
            trials.extend(_run_trial(
                env, poses, kp_fw=kp_fw, level=level, seed=seed,
                steps=math.ceil(args.duration_s / env.step_dt),
                push_tick=math.floor(args.push_at_s / env.step_dt),
                target_wave_rad=math.radians(args.target_wave_deg),
                initial_perturb_rad=math.radians(args.initial_joint_error_deg),
                push_linear_m_s=args.push_linear_m_s,
                push_angular_rad_s=args.push_angular_rad_s,
            ))
    by_name = {pose.name: [] for pose in poses}
    for row in trials:
        by_name[row["candidate"]].append(row)
    results = [{**asdict(pose), **_score(by_name[pose.name], args.duration_s)} for pose in poses]
    return results, trials


def _evaluate_across_gains(
    envs: dict[int, ManagerBasedRlEnv],
    poses: list[Candidate],
    *,
    seeds: list[int],
    push_levels: list[float],
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict]]:
    trials = []
    for kp_fw, env in envs.items():
        _, gain_trials = _evaluate(
            env, poses, kp_fw=kp_fw, seeds=seeds, push_levels=push_levels, args=args,
        )
        for row in gain_trials:
            row["scenario"] = "21_joint_wave_and_optional_body_push"
        trials.extend(gain_trials)
        # A quiet hold is a necessary control: a pose must not score well only
        # because the target waveform happens to counteract its natural fall.
        quiet_args = argparse.Namespace(**vars(args))
        quiet_args.target_wave_deg = 0.0
        quiet_args.initial_joint_error_deg = 0.0
        _, quiet_trials = _evaluate(
            env, poses, kp_fw=kp_fw, seeds=[seeds[0]], push_levels=[0.0], args=quiet_args,
        )
        for row in quiet_trials:
            row["scenario"] = "quiet_hold"
        trials.extend(quiet_trials)
    by_name = {pose.name: [] for pose in poses}
    for row in trials:
        by_name[row["candidate"]].append(row)
    return [
        {**asdict(pose), **_score(by_name[pose.name], args.duration_s)}
        for pose in poses
    ], trials


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--elite-fraction", type=float, default=0.2)
    parser.add_argument("--train-seeds", type=int, default=3,
                        help="Fresh common-random-number seeds per generation")
    parser.add_argument("--heldout-seeds", type=int, default=12)
    parser.add_argument("--test-seeds", type=int, default=12,
                        help="Independent final seeds, never used to choose a HOME")
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--initial-mean", default="-10,0,10", metavar="HIP,ANKLE,SHOULDER")
    parser.add_argument("--initial-sigma", default="5,4,5", metavar="HIP,ANKLE,SHOULDER")
    parser.add_argument("--min-sigma-deg", type=float, default=0.25)
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--push-at-s", type=float, default=1.0)
    parser.add_argument("--push-levels", default="0,1")
    parser.add_argument("--target-wave-deg", type=float, default=1.0)
    parser.add_argument("--initial-joint-error-deg", type=float, default=0.5)
    parser.add_argument("--push-linear-m-s", type=float, default=0.08)
    parser.add_argument("--push-angular-rad-s", type=float, default=0.4)
    parser.add_argument("--kp-values", default="125,900")
    parser.add_argument("--voltage-v", type=float, default=10.8)
    parser.add_argument("--drop-gain-v-per-nm", type=float, default=0.1)
    parser.add_argument("--delay-steps", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/home_search/learned_home.json"))
    return parser.parse_args()


def _three_floats(value: str, label: str) -> np.ndarray:
    result = np.array([float(part) for part in value.split(",")], dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{label} needs exactly three finite comma-separated values")
    return result


def main() -> None:
    args = parse_args()
    if (args.population < 4 or args.generations < 1 or args.train_seeds < 1
            or args.heldout_seeds < 1 or args.test_seeds < 1):
        raise ValueError("Require population >= 4 and positive generation/seed counts")
    if not 0.0 < args.elite_fraction <= 0.5:
        raise ValueError("Require 0 < elite-fraction <= 0.5")
    if not 0.0 < args.push_at_s < args.duration_s:
        raise ValueError("Require 0 < push-at-s < duration-s")
    if min(args.target_wave_deg, args.initial_joint_error_deg, args.push_linear_m_s,
           args.push_angular_rad_s, args.min_sigma_deg) < 0:
        raise ValueError("Disturbance amplitudes and min-sigma must be nonnegative")
    if not 9.0 <= args.voltage_v <= 12.6 or not 0.0 <= args.drop_gain_v_per_nm <= 0.2:
        raise ValueError("Actuator voltage/drop settings exceed the configured range")
    kp_values = [int(value) for value in args.kp_values.split(",")]
    if not kp_values or any(value <= 0 for value in kp_values) or args.delay_steps < 0:
        raise ValueError("Require positive KP values and nonnegative delay")
    push_levels = [float(value) for value in args.push_levels.split(",")]
    if not push_levels or any(not math.isfinite(value) or value < 0 for value in push_levels):
        raise ValueError("push-levels must be finite nonnegative values")
    mean = _three_floats(args.initial_mean, "initial-mean")
    sigma = _three_floats(args.initial_sigma, "initial-sigma")
    if np.any(mean < LOWER) or np.any(mean > UPPER) or np.any(sigma <= 0):
        raise ValueError("Initial mean must be within bounds and sigma must be positive")

    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    rng = np.random.default_rng(args.seed)
    model = mujoco.MjModel.from_xml_path(str(MICROBAN_XML))
    data = mujoco.MjData(model)
    history: list[dict] = []
    finalists: list[tuple[str, np.ndarray]] = [
        ("old_home", OLD_HOME), ("centered_home", CENTERED_HOME),
    ]
    envs = {
        kp_fw: ManagerBasedRlEnv(
            cfg=_configure_static_env(
                args.population, args.duration_s, kp_fw,
                args.voltage_v, args.drop_gain_v_per_nm, args.delay_steps,
            ),
            device=args.device,
        )
        for kp_fw in kp_values
    }
    control_period_s = next(iter(envs.values())).step_dt
    try:
        for generation in range(args.generations):
            samples = np.clip(rng.normal(mean, sigma, (args.population, 3)), LOWER, UPPER)
            samples[0] = OLD_HOME
            samples[1] = CENTERED_HOME
            samples[2] = mean
            poses = [_pose(f"g{generation:03d}_p{index:03d}", angles, model, data)
                     for index, angles in enumerate(samples)]
            seeds = [args.seed + generation * 10000 + index for index in range(args.train_seeds)]
            results, _ = _evaluate_across_gains(
                envs, poses, seeds=seeds, push_levels=push_levels, args=args,
            )
            scores = np.array([row["score"] for row in results])
            elite_count = max(2, math.ceil(args.population * args.elite_fraction))
            elite_indices = np.argsort(scores)[-elite_count:]
            elite = samples[elite_indices]
            mean = 0.3 * mean + 0.7 * elite.mean(axis=0)
            sigma = np.maximum(args.min_sigma_deg, 0.3 * sigma + 0.7 * elite.std(axis=0))
            best_index = int(np.argmax(scores))
            finalists.append((f"generation_{generation:03d}_best", samples[best_index].copy()))
            history.append({
                "generation": generation,
                "seeds": seeds,
                "mean_deg_after_update": mean.tolist(),
                "sigma_deg_after_update": sigma.tolist(),
                "best": results[best_index],
                "candidates": results,
            })
            print(f"[HOME] {generation + 1}/{args.generations} "
                  f"score={scores[best_index]:.4f} "
                  f"angles={samples[best_index].round(3).tolist()} "
                  f"mean={mean.round(3).tolist()}", flush=True)
    finally:
        for env in envs.values():
            env.close()

    # Selection is made on unseen disturbance realizations. Every finalist
    # receives the same trials; a lucky training generation cannot win merely
    # because its sampled perturbations happened to be easy.
    validation_poses = [_pose(name, angles, model, data) for name, angles in finalists]
    envs = {
        kp_fw: ManagerBasedRlEnv(
            cfg=_configure_static_env(
                len(validation_poses), args.duration_s, kp_fw,
                args.voltage_v, args.drop_gain_v_per_nm, args.delay_steps,
            ),
            device=args.device,
        )
        for kp_fw in kp_values
    }
    try:
        heldout = [args.seed + 1_000_000 + index for index in range(args.heldout_seeds)]
        validation, validation_trials = _evaluate_across_gains(
            envs, validation_poses, seeds=heldout, push_levels=push_levels, args=args,
        )
    finally:
        for env in envs.values():
            env.close()
    learned = max(validation[2:], key=lambda row: row["score"])
    best_validated = max(validation, key=lambda row: row["score"])
    test_poses = [
        _pose("old_home", OLD_HOME, model, data),
        _pose("centered_home", CENTERED_HOME, model, data),
        _pose("learned_candidate", np.array([
            learned["hip_deg"], learned["ankle_deg"], learned["shoulder_deg"],
        ]), model, data),
    ]
    test_envs = {
        kp_fw: ManagerBasedRlEnv(
            cfg=_configure_static_env(
                len(test_poses), args.duration_s, kp_fw,
                args.voltage_v, args.drop_gain_v_per_nm, args.delay_steps,
            ),
            device=args.device,
        )
        for kp_fw in kp_values
    }
    try:
        test_seeds = [args.seed + 2_000_000 + index for index in range(args.test_seeds)]
        independent_test, test_trials = _evaluate_across_gains(
            test_envs, test_poses, seeds=test_seeds, push_levels=push_levels, args=args,
        )
    finally:
        for env in test_envs.values():
            env.close()
    report = {
        "metadata": {
            "method": "cross-entropy-method continuous HOME learning with paired 21-joint disturbances",
            "parameters": PARAMETERS,
            "bounds_deg": {name: [float(lower), float(upper)]
                           for name, lower, upper in zip(PARAMETERS, LOWER, UPPER)},
            "fixed_joint_angles": "All HOME angles except symmetric hip/ankle/shoulder pitch retain HOME_FRAME",
            "controller": "shared static position hold; no walking policy or per-HOME controller training",
            "control_period_s": control_period_s,
            "robot_xml_sha256": _digest(MICROBAN_XML),
            "robot_constants_sha256": _digest(Path(microban_constants.__file__)),
            "device": args.device,
            "seed": args.seed,
            "population": args.population,
            "generations": args.generations,
            "elite_fraction": args.elite_fraction,
            "train_seeds_per_generation": args.train_seeds,
            "heldout_seeds": heldout,
            "independent_test_seeds": test_seeds,
            "duration_s": args.duration_s,
            "push_at_s": args.push_at_s,
            "push_levels": push_levels,
            "target_wave_deg": args.target_wave_deg,
            "initial_joint_error_deg": args.initial_joint_error_deg,
            "push_linear_m_s": args.push_linear_m_s,
            "push_angular_rad_s": args.push_angular_rad_s,
            "kp_fw": kp_values,
            "actuator_voltage_v": args.voltage_v,
            "actuator_drop_gain_v_per_nm": args.drop_gain_v_per_nm,
            "actuator_delay_steps": args.delay_steps,
            "fall_definition": "root z < 0.10 m or projected gravity z > -0.5",
            "score": "0.5 mean + 0.5 minimum across shaken, pushed, and quiet trials "
                     "of survival fraction + "
                     "0.1 intact bonus - 0.003 mean tilt degrees",
            "deployment": "simulation only; physical robot HOME and policy unchanged",
        },
        "learned_home": learned,
        "selected_by_validation": best_validated,
        "selected_home_joint_deg": _full_home_joint_deg(best_validated),
        "selection_learned_improves_on_baselines": learned["score"] > max(
            row["score"] for row in validation[:2]
        ),
        "independent_test_score_delta_vs_best_baseline": independent_test[2]["score"] - max(
            row["score"] for row in independent_test[:2]
        ),
        "baselines": validation[:2],
        "validation": validation,
        "validation_trials": validation_trials,
        "independent_test": independent_test,
        "independent_test_trials": test_trials,
        "training_history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[HOME] held-out learned={learned['score']:.4f} "
          f"old={validation[0]['score']:.4f} centered={validation[1]['score']:.4f}")
    print(f"[HOME] independent test learned={independent_test[2]['score']:.4f} "
          f"old={independent_test[0]['score']:.4f} centered={independent_test[1]['score']:.4f}")
    print(f"[HOME] best validated: {best_validated['name']} "
          f"({best_validated['hip_deg']:.3f}, {best_validated['ankle_deg']:.3f}, "
          f"{best_validated['shoulder_deg']:.3f}) deg")
    print(f"[HOME] report: {args.output}")


if __name__ == "__main__":
    main()
