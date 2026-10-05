#!/usr/bin/env python3
"""Re-balance config/home_pose.yaml by moving only the hip and ankle pitches.

Usage (from the training repository root)::

    uv run python config/balance_home_pose.py                 # dry run: before/after table
    uv run python config/balance_home_pose.py --write         # rewrite the 4 pitch values
    uv run python config/balance_home_pose.py --trunk-pitch-deg 10 --write
    uv run python config/balance_home_pose.py --check         # exit 1 unless balanced
    uv run python config/balance_home_pose.py --yaml other_home.yaml

Two unknowns, two equations.  The unknowns are the hip pitch and the ankle
pitch, applied to both legs with the same value (the repository's mirror
convention: pitch joints are equal on the left and right).  Every other joint,
including the knees and arms, and the trunk pitch stay exactly as in the YAML
(or the trunk pitch given with ``--trunk-pitch-deg``).  The equations use the
definitions of ``mjlab_microban.robot.home_pose.analyze_pose`` (MuJoCo FK on
the training robot.xml), the same ones the loader checks:

1. the soles are level at the trunk pitch: ``flat_sole_trunk_pitch_rad`` (the
   root pitch that makes the mean world pitch of the two sole normals zero)
   equals the target trunk pitch;
2. the whole-body COM is over the fore-aft centre of the sole contact area:
   ``com_offset_x`` = COM x - (rearmost + foremost x of the sole collision
   box corners that touch the ground) / 2 = 0.

The solve is a Newton iteration with a central-difference Jacobian, evaluated
with the soles levelled so the contact area stays the full sole (both
residuals are then smooth).  If Newton does not converge, a scan over the
hip pitch range (with the ankle re-levelling the soles at every hip value)
brackets the COM residual and Brent's method finishes the solve, or proves
that the COM cannot reach the sole centre.  The result must lie inside the
MJCF joint range and the training soft limits (0.9 of the range, as
``MICROBAN_ROBOT_CFG``).

``--write`` changes only the hip/ankle pitch values (and ``trunk_pitch_deg``
when ``--trunk-pitch-deg`` differs from the YAML), keeps every comment and
line, writes full float precision, then re-loads the file through the HOME
loader.  An already balanced HOME is left untouched.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mjlab_microban.robot.home_pose import (
    HOME_JOINT_NAMES,
    HOME_POSE_YAML,
    PoseAnalysis,
    analyze_pose,
    home_pose_from_values,
    joint_limits_rad,
    load_home_pose,
    read_home_pose_yaml,
    rewrite_home_pose_yaml,
    validate_home_inputs,
)

SIDES = ("left", "right")
HIP = "hip_pitch"
ANKLE = "ankle_pitch"
BALANCED_JOINTS = tuple(f"{side}_{joint}" for joint in (HIP, ANKLE) for side in SIDES)

# MICROBAN_ROBOT_CFG.articulation.soft_joint_pos_limit_factor: mjlab's soft
# limits are the middle 90 % of the MJCF range (tests check they agree).
SOFT_JOINT_POS_LIMIT_FACTOR = 0.9

# Convergence: both residuals far below the loader's 1e-9 rad flat-sole check
# and the requested 1e-10 rad / 1e-6 mm.
SOLE_TOLERANCE_RAD = 1.0e-13
COM_TOLERANCE_M = 1.0e-13
MAX_NEWTON_ITERATIONS = 60
MAX_NEWTON_STEP_RAD = 0.2
JACOBIAN_STEP_RAD = 1.0e-6
# COM residual weight in the Newton line search (~ COM lever per radian).
COM_SCALE_M = 0.15
HIP_SCAN_POINTS = 181
MAX_TRUNK_PITCH_DEG = 45.0


class BalanceError(ValueError):
    """The HOME cannot be balanced (the message says why)."""


@dataclass(frozen=True)
class BalanceResult:
    path: Path | None
    yaml_trunk_pitch_deg: float
    trunk_pitch_deg: float
    before_deg: Mapping[str, float]
    after_deg: Mapping[str, float]
    before: PoseAnalysis
    after: PoseAnalysis
    changed: bool
    method: str
    iterations: int

    @property
    def updates_deg(self) -> dict[str, float]:
        return {name: self.after_deg[name] for name in BALANCED_JOINTS}

    @property
    def trunk_pitch_changed(self) -> bool:
        return self.trunk_pitch_deg != self.yaml_trunk_pitch_deg


# --------------------------------------------------------------------------
# Residuals


def _pose_rad(
    base_rad: Mapping[str, float], hip: float, ankle: float
) -> dict[str, float]:
    joints = dict(base_rad)
    for side in SIDES:
        joints[f"{side}_{HIP}"] = hip
        joints[f"{side}_{ANKLE}"] = ankle
    return joints


def _levelled(base_rad: Mapping[str, float], hip: float, ankle: float) -> PoseAnalysis:
    """FK at the root pitch that levels the soles (full sole contact area)."""

    return analyze_pose(_pose_rad(base_rad, hip, ankle))


def _residuals(base_rad, hip, ankle, target_rad) -> np.ndarray:
    analysis = _levelled(base_rad, hip, ankle)
    return np.array(
        [analysis.flat_sole_trunk_pitch_rad - target_rad, analysis.com_offset_x],
        dtype=np.float64,
    )


def _converged(residual: np.ndarray) -> bool:
    return (
        abs(residual[0]) <= SOLE_TOLERANCE_RAD and abs(residual[1]) <= COM_TOLERANCE_M
    )


def _merit(residual: np.ndarray) -> float:
    return float(math.hypot(residual[0], residual[1] / COM_SCALE_M))


# --------------------------------------------------------------------------
# Solvers


def _newton(base_rad, hip, ankle, target_rad) -> tuple[float, float, int] | None:
    residual = _residuals(base_rad, hip, ankle, target_rad)
    for iteration in range(1, MAX_NEWTON_ITERATIONS + 1):
        if _converged(residual):
            return hip, ankle, iteration - 1
        step = JACOBIAN_STEP_RAD
        jacobian = np.column_stack(
            [
                (
                    _residuals(base_rad, hip + step, ankle, target_rad)
                    - _residuals(base_rad, hip - step, ankle, target_rad)
                )
                / (2 * step),
                (
                    _residuals(base_rad, hip, ankle + step, target_rad)
                    - _residuals(base_rad, hip, ankle - step, target_rad)
                )
                / (2 * step),
            ]
        )
        try:
            delta = -np.linalg.solve(jacobian, residual)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(delta)):
            return None
        largest = float(np.max(np.abs(delta)))
        if largest > MAX_NEWTON_STEP_RAD:
            delta *= MAX_NEWTON_STEP_RAD / largest
        merit = _merit(residual)
        scale = 1.0
        while True:
            trial_hip, trial_ankle = hip + scale * delta[0], ankle + scale * delta[1]
            trial = _residuals(base_rad, trial_hip, trial_ankle, target_rad)
            if _merit(trial) < merit or _converged(trial) or scale < 1.0e-6:
                break
            scale *= 0.5
        if trial_hip == hip and trial_ankle == ankle:
            # The step vanished in floating point: nothing left to improve.
            return (hip, ankle, iteration) if _converged(trial) else None
        hip, ankle, residual = trial_hip, trial_ankle, trial
    return (hip, ankle, MAX_NEWTON_ITERATIONS) if _converged(residual) else None


def _level_ankle(base_rad, hip, ankle_guess, target_rad, ankle_range) -> float | None:
    """Ankle pitch that levels the soles at ``target_rad`` for a fixed hip pitch."""

    ankle = ankle_guess
    for _ in range(40):
        error = _levelled(base_rad, hip, ankle).flat_sole_trunk_pitch_rad - target_rad
        if abs(error) <= 0.1 * SOLE_TOLERANCE_RAD:
            return ankle
        step = JACOBIAN_STEP_RAD
        slope = (
            _levelled(base_rad, hip, ankle + step).flat_sole_trunk_pitch_rad
            - _levelled(base_rad, hip, ankle - step).flat_sole_trunk_pitch_rad
        ) / (2 * step)
        if slope == 0.0 or not math.isfinite(slope):
            return None
        new_ankle = ankle - error / slope
        if not ankle_range[0] - 0.5 <= new_ankle <= ankle_range[1] + 0.5:
            return None
        if new_ankle == ankle:
            return ankle if abs(error) <= SOLE_TOLERANCE_RAD else None
        ankle = new_ankle
    return None


def _bracketed(base_rad, hip, ankle, target_rad, ranges) -> tuple[float, float, int]:
    """Scan the hip range, bracket the COM residual and finish with Brent."""

    from scipy.optimize import brentq

    hip_range, ankle_range = ranges[HIP], ranges[ANKLE]
    hips = np.linspace(hip_range[0], hip_range[1], HIP_SCAN_POINTS)
    # Walk outwards from the current hip so the ankle guess stays continuous.
    start = int(np.argmin(np.abs(hips - hip)))
    samples: dict[int, tuple[float, float]] = {}
    for direction in (1, -1):
        guess = ankle
        index = start
        while 0 <= index < len(hips):
            levelled = _level_ankle(
                base_rad, float(hips[index]), guess, target_rad, ankle_range
            )
            if levelled is None:
                break
            offset = _levelled(base_rad, float(hips[index]), levelled).com_offset_x
            samples[index] = (levelled, offset)
            guess = levelled
            index += direction
    if not samples:
        raise BalanceError(
            "the ankle pitch cannot level the soles at the requested trunk pitch for any hip pitch"
        )
    indices = sorted(samples)
    offsets_mm = [samples[i][1] * 1e3 for i in indices]
    brackets = [
        (a, b)
        for a, b in itertools.pairwise(indices)
        if b == a + 1 and np.sign(samples[a][1]) != np.sign(samples[b][1])
    ]
    if not brackets:
        raise BalanceError(
            "the COM cannot reach the fore-aft centre of the soles: with the soles level, the "
            f"COM-minus-sole-centre offset stays within [{min(offsets_mm):+.3f}, "
            f"{max(offsets_mm):+.3f}] mm over hip pitch "
            f"[{math.degrees(hips[indices[0]]):+.1f}, {math.degrees(hips[indices[-1]]):+.1f}] deg"
        )
    # Prefer the root closest to the current hip pitch.
    low, high = min(brackets, key=lambda pair: abs(hips[pair[0]] - hip))
    ankle_cache = {"ankle": samples[low][0]}
    evaluations = 0

    def com_offset(trial_hip: float) -> float:
        nonlocal evaluations
        evaluations += 1
        levelled = _level_ankle(
            base_rad, trial_hip, ankle_cache["ankle"], target_rad, ankle_range
        )
        if levelled is None:
            raise BalanceError(
                "lost the level-sole ankle solution inside the hip bracket"
            )
        ankle_cache["ankle"] = levelled
        return _levelled(base_rad, trial_hip, levelled).com_offset_x

    solved_hip = brentq(
        com_offset,
        float(hips[low]),
        float(hips[high]),
        xtol=1.0e-15,
        rtol=4.0 * np.finfo(float).eps,
        maxiter=200,
    )
    solved_ankle = _level_ankle(
        base_rad, solved_hip, ankle_cache["ankle"], target_rad, ankle_range
    )
    if solved_ankle is None:
        raise BalanceError(
            "lost the level-sole ankle solution at the bracketed hip pitch"
        )
    # Polish both unknowns together (removes the inner/outer tolerance mix).
    polished = _newton(base_rad, solved_hip, solved_ankle, target_rad)
    if polished is not None:
        return polished[0], polished[1], evaluations + polished[2]
    return solved_hip, solved_ankle, evaluations


# --------------------------------------------------------------------------
# Public API


def _pitch_ranges() -> dict[str, tuple[float, float]]:
    limits = joint_limits_rad()
    ranges = {}
    for joint in (HIP, ANKLE):
        lower = max(limits[f"{side}_{joint}"][0] for side in SIDES)
        upper = min(limits[f"{side}_{joint}"][1] for side in SIDES)
        ranges[joint] = (lower, upper)
    return ranges


def soft_limits_rad(lower: float, upper: float) -> tuple[float, float]:
    """mjlab soft joint limits: the middle ``SOFT_JOINT_POS_LIMIT_FACTOR`` of the range."""

    middle, half = 0.5 * (lower + upper), 0.5 * (upper - lower)
    return (
        middle - SOFT_JOINT_POS_LIMIT_FACTOR * half,
        middle + SOFT_JOINT_POS_LIMIT_FACTOR * half,
    )


def _check_inputs(joint_pos_deg: Mapping[str, float], trunk_pitch_deg: float) -> None:
    for joint in (HIP, ANKLE):
        left, right = (
            joint_pos_deg.get(f"left_{joint}"),
            joint_pos_deg.get(f"right_{joint}"),
        )
        if left is not None and right is not None and float(left) != float(right):
            raise BalanceError(
                f"left_{joint} ({left!r} deg) and right_{joint} ({right!r} deg) are not "
                "mirror-consistent: pitch joints must be equal on both legs. Make them equal "
                "in the YAML (the balance tool solves one value for both legs)."
            )
    # Every other HOME rule (all 21 joints, numbers, mirror symmetry, MJCF
    # range).  The balanced pitches are checked on the solution instead, so
    # stand them in with 0 here.
    try:
        validate_home_inputs(
            {**joint_pos_deg, **{name: 0.0 for name in BALANCED_JOINTS}},
            trunk_pitch_deg,
        )
    except ValueError as error:
        raise BalanceError(str(error)) from error
    if abs(float(trunk_pitch_deg)) > MAX_TRUNK_PITCH_DEG:
        raise BalanceError(
            f"trunk pitch {trunk_pitch_deg!r} deg is outside +-{MAX_TRUNK_PITCH_DEG} deg"
        )


def balance_joint_pos(
    joint_pos_deg: Mapping[str, float],
    trunk_pitch_deg: float,
    *,
    yaml_trunk_pitch_deg: float | None = None,
    path: Path | None = None,
) -> BalanceResult:
    """Solve hip/ankle pitch (degrees in, degrees out) for a target trunk pitch."""

    if yaml_trunk_pitch_deg is None:
        yaml_trunk_pitch_deg = trunk_pitch_deg
    _check_inputs(joint_pos_deg, trunk_pitch_deg)
    before_deg = {name: float(joint_pos_deg[name]) for name in HOME_JOINT_NAMES}
    base_rad = {name: float(np.deg2rad(value)) for name, value in before_deg.items()}
    target_rad = float(np.deg2rad(float(trunk_pitch_deg)))
    hip0, ankle0 = base_rad[f"left_{HIP}"], base_rad[f"left_{ANKLE}"]
    before = analyze_pose(base_rad, float(np.deg2rad(float(yaml_trunk_pitch_deg))))

    if _converged(_residuals(base_rad, hip0, ankle0, target_rad)):
        after_deg, method, iterations = dict(before_deg), "already balanced", 0
    else:
        ranges = _pitch_ranges()
        solved = _newton(base_rad, hip0, ankle0, target_rad)
        method = "newton"
        if solved is None:
            solved = _bracketed(base_rad, hip0, ankle0, target_rad, ranges)
            method = "scan + brent"
        hip, ankle, iterations = solved
        for joint, value in ((HIP, hip), (ANKLE, ankle)):
            lower, upper = ranges[joint]
            if not lower <= value <= upper:
                raise BalanceError(
                    f"the balanced {joint} {math.degrees(value):+.6f} deg is outside its MJCF "
                    f"range [{math.degrees(lower):+.3f}, {math.degrees(upper):+.3f}] deg"
                )
            soft_lower, soft_upper = soft_limits_rad(lower, upper)
            if not soft_lower <= value <= soft_upper:
                raise BalanceError(
                    f"the balanced {joint} {math.degrees(value):+.6f} deg is outside the training "
                    f"soft limits [{math.degrees(soft_lower):+.3f}, {math.degrees(soft_upper):+.3f}] "
                    f"deg ({SOFT_JOINT_POS_LIMIT_FACTOR} of the MJCF range)"
                )
        after_deg = dict(before_deg)
        hip_deg, ankle_deg = float(np.rad2deg(hip)), float(np.rad2deg(ankle))
        for side in SIDES:
            after_deg[f"{side}_{HIP}"] = hip_deg
            after_deg[f"{side}_{ANKLE}"] = ankle_deg

    # Verify with exactly the loader's numbers: degrees -> radians, analysed at
    # the target trunk pitch (the contact area of a level sole).
    after_rad = {name: float(np.deg2rad(value)) for name, value in after_deg.items()}
    after = analyze_pose(after_rad, target_rad)
    sole_error = abs(after.flat_sole_trunk_pitch_rad - target_rad)
    if sole_error > SOLE_TOLERANCE_RAD or abs(after.com_offset_x) > COM_TOLERANCE_M:
        raise BalanceError(
            f"the solve did not converge: soles {sole_error:.3e} rad from level, COM "
            f"{after.com_offset_x * 1e3:+.3e} mm from the sole centre"
        )
    # Same checks as loading the YAML (symmetry, ranges, flat soles).
    try:
        home_pose_from_values(
            joint_pos_deg=after_deg, trunk_pitch_deg=float(trunk_pitch_deg)
        )
    except ValueError as error:
        raise BalanceError(str(error)) from error
    changed = after_deg != before_deg or float(trunk_pitch_deg) != float(
        yaml_trunk_pitch_deg
    )
    return BalanceResult(
        path=path,
        yaml_trunk_pitch_deg=float(yaml_trunk_pitch_deg),
        trunk_pitch_deg=float(trunk_pitch_deg),
        before_deg=before_deg,
        after_deg=after_deg,
        before=before,
        after=after,
        changed=changed,
        method=method,
        iterations=iterations,
    )


def balance_home_yaml(
    path: Path | str = HOME_POSE_YAML, *, trunk_pitch_deg: float | None = None
) -> BalanceResult:
    """Read a HOME YAML (it need not be balanced yet) and solve it."""

    path = Path(path)
    try:
        document = read_home_pose_yaml(path)
    except ValueError as error:
        raise BalanceError(str(error)) from error
    joint_pos_deg = document["joint_pos_deg"]
    yaml_pitch = document["trunk_pitch_deg"]
    if isinstance(yaml_pitch, bool) or not isinstance(yaml_pitch, (int, float)):
        raise BalanceError(f"{path}: trunk_pitch_deg must be a number")
    target = float(yaml_pitch) if trunk_pitch_deg is None else float(trunk_pitch_deg)
    if not math.isfinite(target):
        raise BalanceError("the trunk pitch must be finite")
    return balance_joint_pos(
        joint_pos_deg,  # type: ignore[arg-type]
        target,
        yaml_trunk_pitch_deg=float(yaml_pitch),
        path=path,
    )


def write_balanced_yaml(result: BalanceResult) -> None:
    """Rewrite the YAML values in place and re-load it through the HOME loader."""

    if result.path is None:
        raise ValueError("the result has no YAML path")
    if not result.changed:
        return
    original = result.path.read_text(encoding="utf-8")
    try:
        rewrite_home_pose_yaml(
            result.path,
            joint_pos_deg=result.updates_deg,
            trunk_pitch_deg=result.trunk_pitch_deg
            if result.trunk_pitch_changed
            else None,
        )
        home = load_home_pose(result.path)
        if dict(home.joint_pos_deg) != dict(result.after_deg):
            raise AssertionError("the rewritten YAML does not hold the balanced values")
        if home.trunk_pitch_deg != result.trunk_pitch_deg:
            raise AssertionError("the rewritten YAML does not hold the trunk pitch")
        if abs(home.analysis.com_offset_x) > COM_TOLERANCE_M:
            raise AssertionError("the re-loaded HOME is not balanced")
    except BaseException:
        result.path.write_text(original, encoding="utf-8")
        raise


# --------------------------------------------------------------------------
# CLI


def _table(result: BalanceResult) -> str:
    before, after = result.before, result.after
    b, a = result.before_deg, result.after_deg

    def deg(value: float) -> str:
        return f"{value:+.15g}"

    def mm(value: float) -> str:
        return f"{value * 1e3:+.6f}"

    def quat(values) -> str:
        return "(" + ", ".join(f"{v:.12f}" for v in values) + ")"

    rows = [
        (
            "left / right hip pitch [deg]",
            f"{deg(b['left_hip_pitch'])} / {deg(b['right_hip_pitch'])}",
            f"{deg(a['left_hip_pitch'])} / {deg(a['right_hip_pitch'])}",
        ),
        (
            "left / right ankle pitch [deg]",
            f"{deg(b['left_ankle_pitch'])} / {deg(b['right_ankle_pitch'])}",
            f"{deg(a['left_ankle_pitch'])} / {deg(a['right_ankle_pitch'])}",
        ),
        (
            "left / right knee [deg]",
            f"{deg(b['left_knee'])} / {deg(b['right_knee'])}",
            f"{deg(a['left_knee'])} / {deg(a['right_knee'])}",
        ),
        (
            "trunk pitch [deg]",
            deg(result.yaml_trunk_pitch_deg),
            deg(result.trunk_pitch_deg),
        ),
        (
            "flat-sole trunk pitch [deg]",
            deg(math.degrees(before.flat_sole_trunk_pitch_rad)),
            deg(math.degrees(after.flat_sole_trunk_pitch_rad)),
        ),
        (
            "sole pitch L / R [deg]",
            " / ".join(f"{math.degrees(v):+.3e}" for v in before.sole_pitch_rad),
            " / ".join(f"{math.degrees(v):+.3e}" for v in after.sole_pitch_rad),
        ),
        ("root z [m]", f"{before.root_pos[2]:.15f}", f"{after.root_pos[2]:.15f}"),
        ("root quat wxyz", quat(before.root_quat_wxyz), quat(after.root_quat_wxyz)),
        (
            "COM - sole centre (x) [mm]",
            f"{before.com_offset_x * 1e3:+.3e}",
            f"{after.com_offset_x * 1e3:+.3e}",
        ),
        ("heel margin [mm]", mm(before.heel_margin_m), mm(after.heel_margin_m)),
        ("toe margin [mm]", mm(before.toe_margin_m), mm(after.toe_margin_m)),
        (
            "sole contact corners",
            str(before.sole_contact_corner_count),
            str(after.sole_contact_corner_count),
        ),
    ]
    width = [
        max(len(row[i]) for row in rows + [("", "before (yaml)", "after")])
        for i in range(3)
    ]
    lines = [
        f"{'':<{width[0]}}  {'before (yaml)':<{width[1]}}  after",
        f"{'-' * width[0]}  {'-' * width[1]}  {'-' * width[2]}",
    ]
    lines += [
        f"{name:<{width[0]}}  {old:<{width[1]}}  {new}" for name, old, new in rows
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Move only hip pitch and ankle pitch (both legs) so the soles are level at the trunk "
            "pitch and the whole-body COM is over the fore-aft centre of the sole contact area."
        )
    )
    parser.add_argument(
        "--yaml", type=Path, default=HOME_POSE_YAML, help="HOME YAML to balance"
    )
    parser.add_argument(
        "--trunk-pitch-deg",
        type=float,
        default=None,
        help="target trunk pitch in degrees (positive leans forward; default: the YAML's)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="rewrite the YAML in place")
    mode.add_argument(
        "--check",
        action="store_true",
        help="exit 1 unless the YAML is already balanced",
    )
    arguments = parser.parse_args(argv)

    try:
        result = balance_home_yaml(
            arguments.yaml, trunk_pitch_deg=arguments.trunk_pitch_deg
        )
    except BalanceError as error:
        print(f"error: cannot balance {arguments.yaml}: {error}", file=sys.stderr)
        return 1
    print(f"HOME yaml: {result.path}")
    print(_table(result))
    print(f"solver: {result.method}, {result.iterations} iterations")
    if not result.changed:
        print("Already balanced: nothing to change.")
        return 0
    if arguments.check:
        print("NOT balanced (run with --write to fix).")
        return 1
    if not arguments.write:
        print(
            "Dry run: re-run with --write to rewrite the hip/ankle pitch values in the YAML."
        )
        return 0
    write_balanced_yaml(result)
    home = load_home_pose(result.path)
    changed = ", ".join(result.updates_deg) + (
        ", trunk_pitch_deg" if result.trunk_pitch_changed else ""
    )
    print(
        f"Wrote {result.path} ({changed}); re-loaded: tag {home.tag}, hash {home.joint_hash}."
    )
    if home.trunk_pitch_deg != 0.0:
        print(
            "Note: this training line accepts only trunk_pitch_deg 0.0 at import; a pitched trunk "
            "needs the home-levelled frames of branch forward-lean-v2."
        )
    print(
        "Next steps (config/README.md):\n"
        "  1. review name/label in the YAML and run: uv run python config/home_pose_tool.py show\n"
        "  2. uv run python config/home_pose_tool.py write-robot --microban-repo ../microban\n"
        "  3. retrain walking, get-up and PICO v12 from scratch at the new HOME, install the\n"
        "     policies in the robot repo, package PICO against it\n"
        "  4. run both test suites and commit both repos"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
