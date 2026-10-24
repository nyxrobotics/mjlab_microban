"""CPU-only regression tests of the IK tracking judgment's evidence and its gate file."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

import torch

from mjlab_microban.schedules import (
    IK_TRACKING_SCHEDULE,
    IK_TRACKING_TOTAL_UPDATES,
)
from mjlab_microban.scripts.evaluate_ik_tracking_v12_checkpoint import (
    LOCOMOTION_MOVING_SCENARIOS,
    locomotion_twist_judgments,
)
from mjlab_microban.scripts.evaluate_ik_tracking_v12_checkpoint import (
    _acceptance as _locomotion_acceptance,
)
from mjlab_microban.scripts.evaluate_ik_tracking_v12_checkpoint import (
    build_parser as build_locomotion_parser,
)
from mjlab_microban.scripts.evaluate_ik_tracking_v12_tracking import (
    build_parser as build_tracking_parser,
)
from mjlab_microban.scripts.evaluate_ik_tracking_v12_tracking import (
    FINAL_PROFILE,
    PUSH_FALL_RATE_MARGIN,
    SEED_COUNT,
    TRACKING_PROFILES,
    ZMP_CENTRE_MAX_M,
    arm_walk_min_speed,
    _acceptance,
    canonical_settings,
    _cfg,
    crouch_target_m,
    feet_on_floor,
    foot_error_limit_m,
    push_fall_rates_by_group,
    required_tracking_check_names,
    required_tracking_profile,
    thresholds,
)
from mjlab_microban.scripts.ik_tracking_v12_bootstrap_gate import (
    ONNX_PARITY_TOLERANCE,
    PRISTINE_PARITY_TOLERANCE,
)
from mjlab_microban.scripts.ik_tracking_v12_scenarios import (
    ENVS_PER_SCENARIO,
    arm_scenarios,
    foot_scenarios,
    push_scenarios,
)
from mjlab_microban.scripts.ik_tracking_v12_stage import (
    _validate_tracking_report,
    create_gate,
    validate_gate,
)
from mjlab_microban.tasks.mdp import MICROBAN_BILATERAL_SITE_ORDER_REVISION
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_ACTION_JOINT_NAMES,
)
from mjlab_microban.tasks.microban_ik_tracking_v12_actor import (
    IK_TRACKING_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION,
    ik_tracking_v12_active_adapter_columns,
)
from mjlab_microban.tasks import microban_ik_tracking_v12_bootstrap
from mjlab_microban.tasks.microban_ik_tracking_v12_bootstrap import (
    portable_bootstrap_artifact_path,
    sha256_file,
)
from mjlab_microban.tasks.microban_ik_tracking_v12_env_cfg import (
    MICROBAN_IK_TRACKING_V12_TRAINING_CONTRACT_VERSION,
    MICROBAN_IK_TRACKING_V13_ARM_OVERLAY_RECIPE_REVISION,
)
from mjlab_microban.tasks.microban_ik_tracking_v12_home_pose import (
    IK_TRACKING_V12_HOME_POSE_INFO_KEY,
    ik_tracking_v12_home_pose_marker,
)
from mjlab_microban.tasks.microban_ik_tracking_v12_runner import (
    BILATERAL_SITE_ORDER_INFO_KEY,
)
from mjlab_microban.ik_tracking_v12_safety import (
    ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD,
)
from mjlab_microban.twist_pass_line import twist_pass_line_record

ROLLOUTS = ENVS_PER_SCENARIO * SEED_COUNT


def _results() -> dict[str, object]:
    """Measurements of a policy that does what every IK tracking check asks."""

    feet = []
    for scenario in foot_scenarios():
        lifted = list(scenario.lifted)
        feet.append({
            "name": scenario.name,
            "foot_goal": [list(goal) for goal in scenario.foot_goal],
            "lifted": lifted,
            "rollouts": ROLLOUTS,
            "falls": 0,
            "lift_median_m": [
                goal[2] if up and not all(lifted) else None for goal, up in zip(scenario.foot_goal, lifted)
            ],
            "air_share_median": [1.0 if up and not all(lifted) else None for up in lifted],
            "trunk_drop_median_m": scenario.foot_goal[0][2] if all(lifted) else None,
            "relative_target_m": math.dist(scenario.foot_goal[0], scenario.foot_goal[1]),
            "relative_error_rms_m": 0.006,
            "support_move_median_m": 0.002 if sum(lifted) == 1 else None,
            "trunk_frame_error_rms_m": 0.006 if all(lifted) else None,
            "touchdowns_per_s": 0.1 if all(lifted) else None,
            "mean_twist": [0.0, 0.0, 0.0] if all(lifted) else None,
            "zmp_distance_median_m": 0.005 if all(lifted) else None,
        })
    # The walker runs standing and walking only.
    push = [
        {"name": s.name, "twist": list(s.twist), "foot_goal": [list(goal) for goal in s.foot_goal],
         "rollouts": ROLLOUTS, "falls": {"pico": 3, **({} if any(s.lifted) else {"walker": 2})},
         "falls_by_direction": {"pico": [0] * 8, **({} if any(s.lifted) else {"walker": [0] * 8})}}
        for s in push_scenarios()
    ]
    arms = []
    for scenario in arm_scenarios():
        axis = next((i for i, value in enumerate(scenario.twist) if value != 0.0), None)
        mean = [0.8 * value for value in scenario.twist]
        arms.append({
            "name": scenario.name, "base": scenario.name.split("/")[0], "arms": scenario.arms,
            "twist": list(scenario.twist), "rollouts": ROLLOUTS, "falls": 0, "mean_twist": mean,
            "signed_speed": None if axis is None else abs(mean[axis]),
            "touchdowns_per_s": 0.1 if axis is None else 3.0,
            "zmp_distance_median_m": 0.005 if axis is None else None,
        })
    safety = {
        "finite": True,
        "raw_action_recurrence": True,
        "maximum_actual_soft_limit_violation_rad": 0.0,
        "hmd_median_target_peak_to_peak_rad": [0.5, 0.5, 0.5],
        "hmd_median_actual_peak_to_peak_rad": [0.4, 0.4, 0.4],
        "arm_observation": {"home_nonzero_steps": 0, "posed_zero_steps": 0},
    }
    return {"feet": feet, "push": push, "arms": arms, "safety": safety}


def _find(records: list[dict[str, object]], name: str) -> dict[str, object]:
    return next(item for item in records if item["name"] == name)


def _locomotion_report(identity: dict[str, object]) -> dict[str, object]:
    commands = {
        "neutral": (0.0, 0.0, 0.0),
        "forward_0p1": (0.1, 0.0, 0.0),
        "forward_0p2": (0.2, 0.0, 0.0),
        "backward_0p1": (-0.1, 0.0, 0.0),
        "backward_0p2": (-0.2, 0.0, 0.0),
        "lateral_left_0p1": (0.0, 0.1, 0.0),
        "lateral_right_0p1": (0.0, -0.1, 0.0),
        "yaw_left_0p5": (0.0, 0.0, 0.5),
        "yaw_right_0p5": (0.0, 0.0, -0.5),
    }
    axis_names = ("vx_m_s", "vy_m_s", "yaw_rad_s")
    results = []
    for name in ("neutral", *LOCOMOTION_MOVING_SCENARIOS):
        twist = commands[name]
        measured = {axis: 0.0 for axis in axis_names}
        response = None
        if name != "neutral":
            # Half the command along it: above every fixed minimum.
            index = next(index for index, value in enumerate(twist) if value != 0.0)
            axis = axis_names[index]
            measured[axis] = 0.5 * twist[index]
            response = {
                "axis": axis,
                "command": twist[index],
                "measured_mean": measured[axis],
                "sign_matches": True,
                "signed_response": abs(measured[axis]),
            }
        results.append(
            {
                "name": name,
                "completed": True,
                "executed_steps": 300,
                "fell": False,
                "nonfinite": None,
                "termination_names": [],
                "maximum_actual_soft_limit_violation_rad": 0.0,
                "raw_action_recurrence_verified_steps": 300,
                "neutral_target_verified_steps": 300,
                "directional_response": response,
                "command": dict(zip(axis_names, twist, strict=True)),
                "measured_velocity_body": {
                    axis: {"count": 250, "mean": mean}
                    for axis, mean in measured.items()
                },
            }
        )
    checks, status = _locomotion_acceptance(results)
    return {
        "schema_version": 1,
        "gate": "microban_teleop_v12_neutral_locomotion_9x300",
        "status": status,
        "checkpoint": identity,
        "settings": {
            "device": "cpu",
            "seed": 42,
            "steps": 300,
            "settle_steps": 50,
            "action_clip": [-math.pi, math.pi],
            "previous_action": "raw_actor_output",
            "policy_observation_width": 81,
        },
        "thresholds": {
            "actual_soft_limit_violation_rad_max": (
                ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD
            ),
            "twist_pass_line": twist_pass_line_record(),
        },
        "checks": checks,
        "twist_judgment": locomotion_twist_judgments(results),
        "results": results,
        "summary": {
            "scenario_count": 9,
            "completed_scenario_count": 9,
            "fall_scenario_count": 0,
            "nonfinite_scenario_count": 0,
            "directionally_correct_scenario_count": 8,
            "directional_scenario_count": 8,
            "minimum_signed_response": 0.025,
            "maximum_actual_soft_limit_violation_rad": 0.0,
        },
    }


def _zero_action_envelope() -> dict[str, object]:
    zero = [0.0] * 18
    summary = {
        "minimum": zero.copy(),
        "maximum": zero.copy(),
        "absolute_maximum": zero.copy(),
    }
    return {
        "joint_names": list(MICROBAN_ACTION_JOINT_NAMES),
        "v12": deepcopy(summary),
        "legacy_source": deepcopy(summary),
        "learned_minus_source": deepcopy(summary),
    }


def _tracking_report(
    identity: dict[str, object], results: dict[str, object] | None = None
) -> dict[str, object]:
    results = _results() if results is None else results
    checks, status = _acceptance(results)
    envelope = _zero_action_envelope()
    envelope.update(
        scenario_count=sum(len(results[part]) for part in ("feet", "push", "arms")), step_count=12000
    )
    return {
        "schema_version": 2,
        "gate": "microban_teleop_v12_tracking",
        "profile": required_tracking_profile(int(identity["completed_updates"])),
        "status": status,
        "checkpoint": identity,
        "settings": canonical_settings(device="cpu", seed=42),
        "thresholds": thresholds(),
        "checks": checks,
        "raw_action_envelope": envelope,
        "runtime_smoke_observations": [],
        "results": results,
    }


def _onnx_report(identity: dict[str, object], onnx_path: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "gate": "microban_teleop_v12_checkpoint_onnx",
        "status": "pass",
        "checkpoint": identity,
        "neutral_legacy_parity": {
            "samples": 10_000,
            "maximum_absolute_error": 0.0,
            "tolerance": PRISTINE_PARITY_TOLERANCE,
            "teleop_only_columns": "exact_zero",
            "residual": "excluded",
        },
        "onnx": {
            "path": portable_bootstrap_artifact_path(onnx_path),
            "sha256": sha256_file(onnx_path),
            "opset": 18,
            "input_shape": [1, 81],
            "output_shape": [1, 18],
            "reference_samples": 64,
            "input_coverage": "deterministic_nonzero_all_81_columns",
            "teleop_only_columns_nonzero": True,
            "reference_evaluator_maximum_absolute_error": 0.0,
            "onnxruntime_cpu_maximum_absolute_error": 0.0,
            "onnxruntime_version": "unit-test",
            "onnxruntime_providers": ["CPUExecutionProvider"],
            "tolerance": ONNX_PARITY_TOLERANCE,
        },
    }


class IkTrackingV12StageTest(unittest.TestCase):
    def test_one_profile_for_the_checkpoint_a_run_ends_with(self) -> None:
        for completed in (IK_TRACKING_SCHEDULE["foot_tighten"] + 1, IK_TRACKING_TOTAL_UPDATES):
            with self.subTest(completed=completed):
                self.assertEqual(required_tracking_profile(completed), FINAL_PROFILE)
        for completed in (IK_TRACKING_SCHEDULE["foot_tighten"], IK_TRACKING_TOTAL_UPDATES + 1, 1):
            with self.subTest(completed=completed), self.assertRaises(ValueError):
                required_tracking_profile(completed)
        self.assertEqual(TRACKING_PROFILES, (FINAL_PROFILE,))
        self.assertEqual(set(_acceptance(_results())[0]), required_tracking_check_names(FINAL_PROFILE))

    def test_the_feet_fail_a_policy_that_keeps_them_down(self) -> None:
        # The v12 IK tracking policy and its pristine start: the lifted foot stays on the
        # floor and the other foot sees it where it was (error = target).
        checks, status = _acceptance(_results())
        self.assertEqual(status, "pass", checks)
        still = _results()
        for item in still["feet"]:
            if all(item["lifted"]):
                item["trunk_drop_median_m"] = 0.0  # no crouch; the feet stay apart as they were
                item["trunk_frame_error_rms_m"] = crouch_target_m(item)
            else:
                item["lift_median_m"] = [0.0 if up else None for up in item["lifted"]]
            item["relative_error_rms_m"] = item["relative_target_m"]
        checks, status = _acceptance(still)
        self.assertEqual(status, "fail")
        self.assertFalse(checks["foot_lift"] or checks["foot_error"])
        self.assertTrue(checks["support_foot_still"])
        # Both feet alone: a policy that does not crouch fails the lift.
        upright = _results()
        _find(upright["feet"], "both_back_right")["trunk_drop_median_m"] = 0.7 * 0.016 - 1.0e-4
        checks = _acceptance(upright)[0]
        self.assertFalse(checks["foot_lift"])
        self.assertTrue(checks["foot_error"])
        # max(0.3 x target, 8 mm): 15.7 mm at a 52 mm corner, 8 mm for both feet.
        self.assertAlmostEqual(foot_error_limit_m(0.0524595), 0.0157379, places=6)
        self.assertEqual(foot_error_limit_m(0.016), 0.008)
        low = _results()
        _find(low["feet"], "left_up40")["lift_median_m"][0] = 0.7 * 0.04 - 1.0e-4
        self.assertFalse(_acceptance(low)[0]["foot_lift"])
        # A foot raised on its toes reaches the height on the floor.
        toes = _results()
        _find(toes["feet"], "right_up20")["air_share_median"][1] = 0.89
        self.assertFalse(_acceptance(toes)[0]["foot_lift"])

    def test_the_judgment_reads_the_feet_left_then_right_as_the_training_does(self) -> None:
        from types import SimpleNamespace

        term = _cfg(0, 10)[0].rewards["lifted_support_feet"]
        sensor = SimpleNamespace(data=SimpleNamespace())
        env = SimpleNamespace(reward_manager=SimpleNamespace(get_term_cfg={"lifted_support_feet": term}.__getitem__),
                              scene=SimpleNamespace(sensors={term.params["sensor_name"]: sensor}))
        # The contact sensor lists the right foot (body foot) first.
        sensor.data.found = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        self.assertEqual(feet_on_floor(env).tolist(), [[False, True], [True, False], [True, True]])
        sensor.data.found = torch.tensor([[[0.0, 2.0], [0.0, 0.0]]])  # per contact slot
        self.assertEqual(feet_on_floor(env).tolist(), [[False, True]])
        # The rollout reads the feet only through it, never the sensor's columns.
        import inspect

        import mjlab_microban.scripts.evaluate_ik_tracking_v12_tracking as judgment

        self.assertNotIn(".found", inspect.getsource(judgment).replace(inspect.getsource(feet_on_floor), ""))
        self.assertEqual(inspect.getsource(judgment._rollout).count("feet_on_floor(env)"), 2)

    def test_a_crouch_goes_down_by_its_height_shifts_by_its_offset_and_stands_still(self) -> None:
        # The trunk drop: at least 0.7 dz.
        for name, drop, lift in (
            ("both_up8", 0.0056 - 1.0e-5, False),
            ("both_up8", 0.0056, True),
            ("both_front_left", 0.0112 - 1.0e-5, False),
        ):
            with self.subTest(name=name, drop=drop):
                results = _results()
                _find(results["feet"], name)["trunk_drop_median_m"] = drop
                self.assertIs(_acceptance(results)[0]["foot_lift"], lift)
        # Each foot from the trunk, as the foot reward measures a crouch:
        # within max(0.3 |offset|, 8 mm) RMS, 8 mm for every judged crouch
        # ((8, 8, 16) mm: 0.3 x 19.6 mm = 5.9 mm).  A crouch that drops its
        # (8, 8) mm shift is 11.3 mm off.
        self.assertAlmostEqual(crouch_target_m(_find(_results()["feet"], "both_front_left")), 0.0196, places=4)
        for name, error, passed in (
            ("both_up16", 0.008, True),
            ("both_up16", 0.008 + 1.0e-5, False),
            ("both_back_right", 0.0113, False),
            ("both_front_left", None, False),
        ):
            with self.subTest(name=name, error=error):
                results = _results()
                _find(results["feet"], name)["trunk_frame_error_rms_m"] = error
                checks = _acceptance(results)[0]
                self.assertIs(checks["foot_error"], passed)
                self.assertTrue(checks["foot_lift"])
        # Crouched on a standing command: no stepping and no drift (as J3).
        stepping = _results()
        _find(stepping["feet"], "both_up16")["touchdowns_per_s"] = 0.6
        self.assertFalse(_acceptance(stepping)[0]["standing_still"])
        drifting = _results()
        _find(drifting["feet"], "both_back_right")["mean_twist"] = [0.0, 0.06, 0.0]
        self.assertFalse(_acceptance(drifting)[0]["standing_still"])
        fell = _results()
        _find(fell["feet"], "both_back_right")["falls"] = 1
        self.assertFalse(_acceptance(fell)[0]["no_falls_without_push"])

    def test_standing_keeps_the_zmp_between_the_soles(self) -> None:
        # A quarter of HOME's sole length (62 mm): half the margin from the
        # soles' midpoint to the toe or heel.
        self.assertAlmostEqual(ZMP_CENTRE_MAX_M, 0.0155, delta=0.0002)
        self.assertEqual(thresholds()["zmp_centre_max_m"], ZMP_CENTRE_MAX_M)
        for part, name in (("feet", "both_up8"), ("feet", "both_front_left"), ("arms", "neutral/arms_home"),
                           ("arms", "neutral/arms_raised"), ("arms", "neutral/arms_moving")):
            for distance, passed in ((ZMP_CENTRE_MAX_M, True), (ZMP_CENTRE_MAX_M + 1.0e-4, False), (None, False)):
                with self.subTest(name=name, distance=distance):
                    results = _results()
                    _find(results[part], name)["zmp_distance_median_m"] = distance
                    checks = _acceptance(results)[0]
                    self.assertIs(checks["standing_still"], passed)
                    self.assertTrue(checks["walking_with_arms"])
        # Not judged on a single-foot target or a walking command.
        self.assertTrue(all(item["zmp_distance_median_m"] is None for item in _results()["feet"]
                            if not all(item["lifted"])))

    def test_the_support_foot_stays(self) -> None:
        moved = _results()
        _find(moved["feet"], "right_front_in")["support_move_median_m"] = 0.011
        self.assertFalse(_acceptance(moved)[0]["support_foot_still"])
        stepping = _results()
        _find(stepping["feet"], "both_front_left")["relative_error_rms_m"] = 0.009
        self.assertFalse(_acceptance(stepping)[0]["foot_error"])

    def test_pushes_are_compared_with_the_walker(self) -> None:
        results = _results()
        for item in results["push"]:
            item["falls"] = {"pico": 0, "walker": 0}
        _find(results["push"], "neutral")["falls"]["pico"] = 57  # 57/1152 < 5 points
        self.assertTrue(_acceptance(results)[0]["push_falls"])
        _find(results["push"], "neutral")["falls"]["pico"] = 58
        self.assertFalse(_acceptance(results)[0]["push_falls"])
        _find(results["push"], "forward_0p1")["falls"]["walker"] = 10
        self.assertTrue(_acceptance(results)[0]["push_falls"])
        # The crouches on their own, against the walker standing and walking
        # (the walker is not run crouched).
        crouched = [s.name for s in push_scenarios() if any(any(goal) for goal in s.foot_goal)]
        self.assertEqual(crouched, ["both_up8", "both_up16", "both_front_left", "both_back_right"])
        results = _results()
        for item in results["push"]:
            item["falls"] = {"pico": 0} if item["name"] in crouched else {"pico": 0, "walker": 0}
        _find(results["push"], "both_up16")["falls"]["pico"] = 25  # 25/512 < 5 points
        self.assertTrue(_acceptance(results)[0]["push_falls"])
        _find(results["push"], "both_up16")["falls"]["pico"] = 26
        self.assertFalse(_acceptance(results)[0]["push_falls"])
        _find(results["push"], "forward_0p2")["falls"]["walker"] = 12  # 12/1152: 1 point more
        self.assertTrue(_acceptance(results)[0]["push_falls"])
        self.assertEqual(push_fall_rates_by_group(results["push"]),
                         {"standing_and_walking": {"pico": 0.0, "walker": 12 / 1152}, "crouch": {"pico": 26 / 512}})
        # Standing and walking falls of the IK tracking policy do not hide behind its crouches.
        results = _results()
        for item in results["push"]:
            item["falls"] = {"pico": 0} if item["name"] in crouched else {"pico": 0, "walker": 0}
        _find(results["push"], "neutral")["falls"]["pico"] = 58
        self.assertFalse(_acceptance(results)[0]["push_falls"])
        # A judged floor's own margin (lines.push_fall_rate_margin, --push-fall-rate-margin; the soft plate's 8
        # points) in place of the 5 points, recorded with the report's thresholds.
        results = _results()
        for item in results["push"]:
            item["falls"] = {"pico": 0} if item["name"] in crouched else {"pico": 0, "walker": 0}
        _find(results["push"], "both_up16")["falls"]["pico"] = 40  # 40/512 < 8 points
        self.assertFalse(_acceptance(results)[0]["push_falls"])
        self.assertTrue(_acceptance(results, 0.08)[0]["push_falls"])
        _find(results["push"], "both_up16")["falls"]["pico"] = 41
        self.assertFalse(_acceptance(results, 0.08)[0]["push_falls"])
        self.assertEqual((thresholds()["push_fall_rate_margin"], thresholds(0.08)["push_fall_rate_margin"]),
                         (PUSH_FALL_RATE_MARGIN, 0.08))
        self.assertEqual([build_tracking_parser().parse_args(["model.pt", *flag]).push_fall_rate_margin
                          for flag in ([], ["--push-fall-rate-margin", "0.08"])], [PUSH_FALL_RATE_MARGIN, 0.08])

    def test_a_judged_floor_may_loosen_the_9x300_direction(self) -> None:
        # A floor's twist_minimum_scale (--twist-minimum-scale): 0.3 of every command is below the minimums and
        # above half of them.  The rigid floor's records stay as the stage gate recomputes them.
        results = _locomotion_report({"sha256": "a" * 64, "iteration": 0, "completed_updates": 1})["results"]
        for result in results[1:]:
            response = result["directional_response"]
            result["measured_velocity_body"][response["axis"]]["mean"] = 0.3 * response["command"]
        self.assertFalse(_locomotion_acceptance(results)[0]["twist_direction"])
        self.assertTrue(_locomotion_acceptance(results, 0.5)[0]["twist_direction"])
        self.assertEqual({item["minimum_scale"] for item in locomotion_twist_judgments(results, 0.5).values()}, {0.5})
        self.assertFalse(any("minimum_scale" in item for item in locomotion_twist_judgments(results).values()))
        self.assertEqual([build_locomotion_parser().parse_args(["model.pt", *flag]).twist_minimum_scale
                          for flag in ([], ["--twist-minimum-scale", "0.5"])], [1.0, 0.5])

    def test_standing_still_with_the_head_and_arms_moving(self) -> None:
        stepping = _results()
        _find(stepping["arms"], "neutral/arms_moving")["touchdowns_per_s"] = 0.6
        self.assertFalse(_acceptance(stepping)[0]["standing_still"])
        drifting = _results()
        _find(drifting["arms"], "neutral/arms_raised")["mean_twist"] = [0.06, 0.0, 0.0]
        self.assertFalse(_acceptance(drifting)[0]["standing_still"])
        reaching = _results()
        _find(reaching["arms"], "neutral/arms_reach_right")["touchdowns_per_s"] = 0.6
        self.assertFalse(_acceptance(reaching)[0]["standing_still"])

    def test_walking_with_the_arms_moved_walks_along_the_command(self) -> None:
        # Command 0.1 m/s, 0.08 m/s with the arms at HOME: with the arms moved
        # it may walk slower, down to 20 % of the command.
        self.assertAlmostEqual(arm_walk_min_speed(0.08, 0.1), 0.02)
        self.assertAlmostEqual(arm_walk_min_speed(0.01, 0.1), 0.01)
        for mode, speed, passed in (
            ("raised", 0.0205, True),
            ("raised", 0.0195, False),
            ("moving", 0.11, True),
            ("moving", -0.01, False),
        ):
            with self.subTest(mode=mode, speed=speed):
                results = _results()
                _find(results["arms"], f"forward_0p1/arms_{mode}")["signed_speed"] = speed
                self.assertIs(_acceptance(results)[0]["walking_with_arms"], passed)
        # A walker that does not turn at HOME does not make the arms fail.
        stuck = _results()
        for mode in ("home", "raised", "moving"):
            _find(stuck["arms"], f"yaw_left_0p5/arms_{mode}")["signed_speed"] = 0.0
        self.assertTrue(_acceptance(stuck)[0]["walking_with_arms"])
        missing = _results()
        _find(missing["arms"], "yaw_left_0p5/arms_home")["signed_speed"] = None
        self.assertFalse(_acceptance(missing)[0]["walking_with_arms"])
        fell = _results()
        _find(fell["arms"], "lateral_left_0p1/arms_moving")["falls"] = 1
        self.assertFalse(_acceptance(fell)[0]["no_falls_without_push"])

    def test_the_gate_judges_the_report_under_the_final_profile(self) -> None:
        identity = {
            "sha256": "a" * 64,
            "iteration": IK_TRACKING_TOTAL_UPDATES - 1,
            "completed_updates": IK_TRACKING_TOTAL_UPDATES,
        }
        report = _tracking_report(identity)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(_validate_tracking_report(report, identity), FINAL_PROFILE)
        still = _results()
        _find(still["feet"], "left_up20")["lift_median_m"][0] = 0.0
        with self.assertRaises(ValueError):
            _validate_tracking_report(_tracking_report(identity, still), identity)
        relabelled = deepcopy(report)
        relabelled["profile"] = "full_body_reachable_performance_perturbation_v2_deployed_accuracy_v1"
        with self.assertRaisesRegex(ValueError, "schema/profile/status drifted"):
            _validate_tracking_report(relabelled, identity)
        for key, value in (("seeds", [101, 102]), ("envs_per_scenario", 1)):
            changed = deepcopy(report)
            changed["settings"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                _validate_tracking_report(changed, identity)
        forged = deepcopy(report)
        _find(forged["results"]["arms"], "forward_0p2/arms_raised")["signed_speed"] = 0.0
        with self.assertRaisesRegex(ValueError, "checks do not match"):
            _validate_tracking_report(forged, identity)

    def test_final_gate_records_the_final_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / f"model_{IK_TRACKING_TOTAL_UPDATES - 1}.pt"
            infos = {
                "microban_teleop_training_contract_version": (
                    MICROBAN_IK_TRACKING_V12_TRAINING_CONTRACT_VERSION
                ),
                "microban_teleop_recipe_revision": (
                    MICROBAN_IK_TRACKING_V13_ARM_OVERLAY_RECIPE_REVISION
                ),
                BILATERAL_SITE_ORDER_INFO_KEY: MICROBAN_BILATERAL_SITE_ORDER_REVISION,
                IK_TRACKING_V12_HOME_POSE_INFO_KEY: ik_tracking_v12_home_pose_marker(),
                "adapter_gradient_schedule_revision": (
                    IK_TRACKING_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION
                ),
                "active_actor_columns_at_save": list(
                    ik_tracking_v12_active_adapter_columns(IK_TRACKING_TOTAL_UPDATES * 24)
                ),
                "env_state": {"common_step_counter": IK_TRACKING_TOTAL_UPDATES * 24},
            }
            torch.save({"iter": IK_TRACKING_TOTAL_UPDATES - 1, "infos": infos}, checkpoint)
            identity = {
                "sha256": sha256_file(checkpoint),
                "iteration": IK_TRACKING_TOTAL_UPDATES - 1,
                "completed_updates": IK_TRACKING_TOTAL_UPDATES,
            }
            paths = {
                "locomotion_report": root / "locomotion.json",
                "tracking_report": root / "tracking.json",
                "onnx_report": root / "onnx.json",
            }
            onnx_path = root / "policy.onnx"
            onnx_path.write_bytes(b"unit-test-onnx")
            paths["locomotion_report"].write_text(json.dumps(_locomotion_report(identity)))
            paths["tracking_report"].write_text(
                json.dumps(_tracking_report(identity))
            )
            paths["onnx_report"].write_text(json.dumps(_onnx_report(identity, onnx_path)))
            gate = create_gate(checkpoint=checkpoint, **paths)
            self.assertEqual(gate["tracking_profile"], FINAL_PROFILE)
            gate_path = root / "gate.json"
            gate_path.write_text(json.dumps(gate))
            self.assertEqual(validate_gate(gate_path, checkpoint), gate)
            tampered = deepcopy(gate)
            tampered["tracking_profile"] = "whole_body_reachable_performance_v2_deployed_accuracy_v1"
            gate_path.write_text(json.dumps(tampered))
            with self.assertRaises(ValueError):
                validate_gate(gate_path, checkpoint)

    def test_only_the_last_checkpoint_of_a_run_is_gated(self) -> None:
        for completed, accepted in ((IK_TRACKING_TOTAL_UPDATES, True), (IK_TRACKING_TOTAL_UPDATES - 1, False)):
            infos = {
                "microban_teleop_training_contract_version": (
                    MICROBAN_IK_TRACKING_V12_TRAINING_CONTRACT_VERSION
                ),
                "microban_teleop_recipe_revision": (
                    MICROBAN_IK_TRACKING_V13_ARM_OVERLAY_RECIPE_REVISION
                ),
                BILATERAL_SITE_ORDER_INFO_KEY: MICROBAN_BILATERAL_SITE_ORDER_REVISION,
                IK_TRACKING_V12_HOME_POSE_INFO_KEY: ik_tracking_v12_home_pose_marker(),
                "adapter_gradient_schedule_revision": (
                    IK_TRACKING_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION
                ),
                "active_actor_columns_at_save": list(
                    ik_tracking_v12_active_adapter_columns(completed * 24)
                ),
                "env_state": {"common_step_counter": completed * 24},
            }
            with self.subTest(completed=completed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                onnx_path = root / "policy.onnx"
                onnx_path.write_bytes(b"unit-test-onnx")
                checkpoint = root / f"model_{completed - 1}.pt"
                torch.save({"iter": completed - 1, "infos": infos}, checkpoint)
                identity = {
                    "sha256": sha256_file(checkpoint),
                    "iteration": completed - 1,
                    "completed_updates": completed,
                }
                reports = {
                    "locomotion_report": root / "locomotion.json",
                    "tracking_report": root / "tracking.json",
                    "onnx_report": root / "onnx.json",
                }
                reports["locomotion_report"].write_text(json.dumps(_locomotion_report(identity)))
                reports["tracking_report"].write_text(
                    json.dumps(_tracking_report(identity))
                )
                reports["onnx_report"].write_text(json.dumps(_onnx_report(identity, onnx_path)))
                if not accepted:
                    with self.assertRaisesRegex(ValueError, "An IK tracking run ends at update"):
                        create_gate(checkpoint=checkpoint, **reports)
                    continue
                gate = create_gate(checkpoint=checkpoint, **reports)
                self.assertEqual(gate["schema_version"], 3)
                self.assertEqual(gate["tracking_profile"], FINAL_PROFILE)
                gate_path = root / "gate.json"
                gate_path.write_text(json.dumps(gate))
                self.assertEqual(validate_gate(gate_path, checkpoint), gate)

    def test_gate_evidence_is_hash_bound_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / f"model_{IK_TRACKING_TOTAL_UPDATES - 1}.pt"
            infos = {
                "microban_teleop_training_contract_version": (
                    MICROBAN_IK_TRACKING_V12_TRAINING_CONTRACT_VERSION
                ),
                "microban_teleop_recipe_revision": (
                    MICROBAN_IK_TRACKING_V13_ARM_OVERLAY_RECIPE_REVISION
                ),
                BILATERAL_SITE_ORDER_INFO_KEY: MICROBAN_BILATERAL_SITE_ORDER_REVISION,
                IK_TRACKING_V12_HOME_POSE_INFO_KEY: ik_tracking_v12_home_pose_marker(),
                "adapter_gradient_schedule_revision": (
                    IK_TRACKING_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION
                ),
                "active_actor_columns_at_save": list(
                    ik_tracking_v12_active_adapter_columns(IK_TRACKING_TOTAL_UPDATES * 24)
                ),
                "env_state": {"common_step_counter": IK_TRACKING_TOTAL_UPDATES * 24},
            }
            torch.save({"iter": IK_TRACKING_TOTAL_UPDATES - 1, "infos": infos}, checkpoint)
            checkpoint_sha = sha256_file(checkpoint)
            identity = {
                "sha256": checkpoint_sha,
                "iteration": IK_TRACKING_TOTAL_UPDATES - 1,
                "completed_updates": IK_TRACKING_TOTAL_UPDATES,
            }
            locomotion_path = root / "locomotion.json"
            tracking_path = root / "tracking.json"
            onnx_report_path = root / "onnx.json"
            onnx_path = root / "policy.onnx"
            onnx_path.write_bytes(b"unit-test-onnx")
            locomotion = _locomotion_report(identity)
            tracking = _tracking_report(identity)
            onnx = _onnx_report(identity, onnx_path)
            locomotion_path.write_text(json.dumps(locomotion))
            tracking_path.write_text(json.dumps(tracking))
            onnx_report_path.write_text(json.dumps(onnx))
            gate = create_gate(
                checkpoint=checkpoint,
                locomotion_report=locomotion_path,
                tracking_report=tracking_path,
                onnx_report=onnx_report_path,
            )
            self.assertEqual(gate["schema_version"], 3)
            gate_path = root / "gate.json"
            gate_path.write_text(json.dumps(gate))
            self.assertEqual(validate_gate(gate_path, checkpoint), gate)

            corruptions = (
                (locomotion_path, locomotion, ("checks",), {}),
                (tracking_path, tracking, ("results",), []),
                (tracking_path, tracking, ("results", "feet"), []),
                (tracking_path, tracking, ("results", "safety", "finite"), False),
                (tracking_path, tracking, ("results", "feet", 0, "lift_median_m", 0), 0.0),
                (tracking_path, tracking, ("results", "feet", 0, "foot_goal", 0, 2), 0.0),
                (
                    onnx_report_path,
                    onnx,
                    ("onnx", "onnxruntime_cpu_maximum_absolute_error"),
                    ONNX_PARITY_TOLERANCE * 2.0,
                ),
                (
                    onnx_report_path,
                    onnx,
                    ("neutral_legacy_parity", "maximum_absolute_error"),
                    -1.0,
                ),
                (
                    onnx_report_path,
                    onnx,
                    ("onnx", "reference_evaluator_maximum_absolute_error"),
                    -1.0,
                ),
            )
            for path, original, keys, replacement in corruptions:
                with self.subTest(keys=keys):
                    corrupted = deepcopy(original)
                    target = corrupted
                    for key in keys[:-1]:
                        target = target[key]
                    target[keys[-1]] = replacement
                    path.write_text(json.dumps(corrupted))
                    with self.assertRaises(ValueError):
                        create_gate(
                            checkpoint=checkpoint,
                            locomotion_report=locomotion_path,
                            tracking_report=tracking_path,
                            onnx_report=onnx_report_path,
                        )
                    locomotion_path.write_text(json.dumps(locomotion))
                    tracking_path.write_text(json.dumps(tracking))
                    onnx_report_path.write_text(json.dumps(onnx))


APPROVAL = "User approved installing this model with these tracking checks failing"


def _final_checkpoint(root: Path) -> tuple[Path, dict[str, object]]:
    checkpoint = root / f"model_{IK_TRACKING_TOTAL_UPDATES - 1}.pt"
    infos = {
        "microban_teleop_training_contract_version": MICROBAN_IK_TRACKING_V12_TRAINING_CONTRACT_VERSION,
        "microban_teleop_recipe_revision": MICROBAN_IK_TRACKING_V13_ARM_OVERLAY_RECIPE_REVISION,
        BILATERAL_SITE_ORDER_INFO_KEY: MICROBAN_BILATERAL_SITE_ORDER_REVISION,
        IK_TRACKING_V12_HOME_POSE_INFO_KEY: ik_tracking_v12_home_pose_marker(),
        "adapter_gradient_schedule_revision": IK_TRACKING_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION,
        "active_actor_columns_at_save": list(ik_tracking_v12_active_adapter_columns(IK_TRACKING_TOTAL_UPDATES * 24)),
        "env_state": {"common_step_counter": IK_TRACKING_TOTAL_UPDATES * 24},
    }
    torch.save({"iter": IK_TRACKING_TOTAL_UPDATES - 1, "infos": infos}, checkpoint)
    identity = {
        "sha256": sha256_file(checkpoint),
        "iteration": IK_TRACKING_TOTAL_UPDATES - 1,
        "completed_updates": IK_TRACKING_TOTAL_UPDATES,
    }
    return checkpoint, identity


def _write_reports(root: Path, identity: dict[str, object], results: dict[str, object]) -> dict[str, Path]:
    paths = {
        "locomotion_report": root / "locomotion.json",
        "tracking_report": root / "tracking.json",
        "onnx_report": root / "onnx.json",
    }
    onnx_path = root / "policy.onnx"
    onnx_path.write_bytes(b"unit-test-onnx")
    paths["locomotion_report"].write_text(json.dumps(_locomotion_report(identity)))
    paths["tracking_report"].write_text(json.dumps(_tracking_report(identity, results)))
    paths["onnx_report"].write_text(json.dumps(_onnx_report(identity, onnx_path)))
    return paths


def _failing_behaviour() -> dict[str, object]:
    """A support foot that steps, the ZMP off-centre in a crouch and
    touchdowns while standing with raised arms."""

    results = _results()
    _find(results["feet"], "left_back_out")["support_move_median_m"] = 0.029
    _find(results["feet"], "both_up16")["zmp_distance_median_m"] = 0.02
    _find(results["arms"], "neutral/arms_raised")["touchdowns_per_s"] = 0.95
    return results


class TrackingWaiverTest(unittest.TestCase):
    def test_a_normal_gate_carries_no_waiver(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, identity = _final_checkpoint(root)
            gate = create_gate(checkpoint=checkpoint, **_write_reports(root, identity, _results()))
            self.assertEqual(gate["status"], "pass")
            self.assertNotIn("tracking_waiver", gate)
            with self.assertRaisesRegex(ValueError, "waiver note needs"):
                create_gate(checkpoint=checkpoint, waiver_note=APPROVAL,
                            **_write_reports(root, identity, _results()))
            with self.assertRaisesRegex(ValueError, "status drifted"):
                create_gate(checkpoint=checkpoint, **_write_reports(root, identity, _failing_behaviour()))

    def test_waived_behaviour_checks_are_recorded_with_their_measurements(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, identity = _final_checkpoint(root)
            paths = _write_reports(root, identity, _failing_behaviour())
            waived = ("support_foot_still", "standing_still")
            gate = create_gate(checkpoint=checkpoint, waive_tracking_checks=waived,
                               waiver_note=APPROVAL, **paths)
            self.assertEqual(gate["status"], "pass_with_waiver")
            self.assertEqual(gate["tracking_waiver"], {
                "approval": APPROVAL,
                "checks": {
                    "standing_still": [{
                        "scenario": "both_up16", "quantity": "zmp_distance_median_m",
                        "measured": 0.02, "limit": ZMP_CENTRE_MAX_M, "rule": "at_most",
                    }, {
                        "scenario": "neutral/arms_raised", "quantity": "touchdowns_per_s",
                        "measured": 0.95, "limit": 0.5, "rule": "at_most",
                    }],
                    "support_foot_still": [{
                        "scenario": "left_back_out", "quantity": "support_move_median_m",
                        "measured": 0.029, "limit": 0.01, "rule": "at_most",
                    }],
                },
            })
            gate_path = root / "gate.json"
            gate_path.write_text(json.dumps(gate))
            self.assertEqual(validate_gate(gate_path, checkpoint), gate)
            for keys, value in (
                (("tracking_waiver", "checks", "standing_still", 0, "limit"), 0.9),
                (("tracking_waiver", "checks", "standing_still", 0, "measured"), 0.6),
                (("status",), "pass"),
            ):
                with self.subTest(keys=keys):
                    tampered = deepcopy(gate)
                    target = tampered
                    for key in keys[:-1]:
                        target = target[key]
                    target[keys[-1]] = value
                    gate_path.write_text(json.dumps(tampered))
                    with self.assertRaises(ValueError):
                        validate_gate(gate_path, checkpoint)
            dropped = deepcopy(gate)
            del dropped["tracking_waiver"]
            gate_path.write_text(json.dumps(dropped))
            with self.assertRaises(ValueError):
                validate_gate(gate_path, checkpoint)
            for names, note, message in (
                (("support_foot_still",), APPROVAL, "exactly the failing"),
                ((*waived, "foot_error"), APPROVAL, "exactly the failing"),
                (waived, None, "approval"),
                (waived, "  ", "approval"),
            ):
                with self.subTest(names=names, note=note), self.assertRaisesRegex(ValueError, message):
                    create_gate(checkpoint=checkpoint, waive_tracking_checks=names, waiver_note=note, **paths)

    def test_safety_checks_cannot_be_waived(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, identity = _final_checkpoint(root)
            results = _failing_behaviour()
            results["safety"]["maximum_actual_soft_limit_violation_rad"] = 0.5
            paths = _write_reports(root, identity, results)
            for names in (
                ("support_foot_still", "standing_still"),
                ("support_foot_still", "standing_still", "actual_soft_limits"),
            ):
                with self.subTest(names=names), self.assertRaisesRegex(ValueError, "Safety checks"):
                    create_gate(checkpoint=checkpoint, waive_tracking_checks=names,
                                waiver_note=APPROVAL, **paths)
            paths = _write_reports(root, identity, _failing_behaviour())
            with self.assertRaisesRegex(ValueError, "Safety checks"):
                create_gate(checkpoint=checkpoint,
                            waive_tracking_checks=("support_foot_still", "standing_still", "finite"),
                            waiver_note=APPROVAL, **paths)


class PortableEvidenceTest(unittest.TestCase):
    def test_a_gate_still_validates_after_the_run_is_copied_to_another_clone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first", Path(directory) / "second"
            run = Path("artifacts/ik_tracking_release/run")
            (first / run).mkdir(parents=True)
            with mock.patch.object(microban_ik_tracking_v12_bootstrap, "_PROJECT_ROOT", first):
                checkpoint, identity = _final_checkpoint(first / run)
                paths = _write_reports(first / run, identity, _failing_behaviour())
                self.assertTrue(
                    json.loads(paths["onnx_report"].read_text())["onnx"]["path"].startswith("repo://")
                )
                gate = create_gate(checkpoint=checkpoint,
                                   waive_tracking_checks=("support_foot_still", "standing_still"),
                                   waiver_note=APPROVAL, **paths)
                (first / run / "gate.json").write_text(json.dumps(gate))
            shutil.copytree(first, second)
            shutil.rmtree(first)
            with mock.patch.object(microban_ik_tracking_v12_bootstrap, "_PROJECT_ROOT", second):
                copied = second / run
                self.assertEqual(validate_gate(copied / "gate.json", copied / checkpoint.name), gate)


if __name__ == "__main__":
    unittest.main()
