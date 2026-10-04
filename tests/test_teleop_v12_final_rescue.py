"""CPU tests for the 14900->14999 final-scenario rescue sampler and marker."""

from __future__ import annotations

import math

import pytest
import torch

from mjlab_microban.scripts.evaluate_teleop_v12_tracking import (
    FINAL_DEPLOYED_ACCURACY_PROFILE,
    _scenarios,
)
from mjlab_microban.tasks.microban_teleop_v12_final_rescue import (
    MICROBAN_TELEOP_V12_FINAL_RESCUE_MIXES,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_OPTIMIZER_STEP,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_SCENARIOS,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_OPTIMIZER_STEP,
    _FinalRescuePatternState,
    final_rescue_marker,
    final_rescue_mix_probabilities,
    final_rescue_scenario_commands,
    make_microban_teleop_v12_final_rescue_env_cfg,
    validate_final_rescue_lineage_marker,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _marker(**overrides):
    values = {
        "parent_checkpoint_sha256": SHA_A,
        "parent_tracking_report_sha256": SHA_B,
        "parent_failed_checks": ["foot_tracking_rms"],
        "failed_gate_checkpoint_sha256": SHA_C,
        "failed_gate_tracking_report_sha256": SHA_D,
        "failed_gate_failed_checks": [
            "foot_tracking_rms",
            "hand_tracking_p95",
            "hand_tracking_rms",
        ],
        "inherited_corner_rescue_marker_sha256": None,
        "sampler_mix": "v1",
    }
    values.update(overrides)
    return final_rescue_marker(**values)


def test_clock_constants_match_ppo_schedule():
    assert MICROBAN_TELEOP_V12_FINAL_RESCUE_PARENT_OPTIMIZER_STEP == 298_020
    assert MICROBAN_TELEOP_V12_FINAL_RESCUE_TARGET_OPTIMIZER_STEP == 300_000


def test_scenario_commands_mirror_the_final_evaluator():
    commands = final_rescue_scenario_commands()
    evaluator = {item.name: item for item in _scenarios(FINAL_DEPLOYED_ACCURACY_PROFILE)}
    assert tuple(commands) == MICROBAN_TELEOP_V12_FINAL_RESCUE_SCENARIOS
    for name, command in commands.items():
        scenario = evaluator[name]
        assert command["twist"] == list(scenario.twist)
        assert command["foot_target"] == [list(xyz) for xyz in scenario.foot_target]
        assert command["hand_target"] == [list(xyz) for xyz in scenario.hand_target]
        assert command["hand_active"] == list(scenario.hand_active) == [True, True]


@pytest.mark.parametrize("mix", sorted(MICROBAN_TELEOP_V12_FINAL_RESCUE_MIXES))
def test_mix_keeps_ten_percent_ordinary(mix):
    probabilities = final_rescue_mix_probabilities(mix)
    assert probabilities["ordinary"] == 0.10
    assert math.isclose(sum(probabilities.values()), 1.0)


def test_pattern_state_draws_once_per_reset_event_in_any_term_order():
    torch.manual_seed(0)
    state = _FinalRescuePatternState(num_envs=4096, device="cpu", mix="v1")
    ids = torch.arange(4096)
    initial = state.pattern.clone()
    for term in ("twist", "foot_target", "hand_target"):
        state.on_reset(term, ids)
    assert torch.equal(state.pattern, initial)
    # Second reset event, terms in a different order: one redraw shared by all.
    state.on_reset("hand_target", ids[:10])
    drawn = state.pattern[:10].clone()
    state.on_reset("twist", ids[:10])
    state.on_reset("foot_target", ids[:10])
    assert torch.equal(state.pattern[:10], drawn)
    assert torch.equal(state.generation[:10], torch.ones(10, dtype=torch.long))
    assert torch.equal(state.generation[10:], torch.zeros(4086, dtype=torch.long))
    counts = torch.bincount(initial, minlength=3).double() / 4096
    assert abs(float(counts[0]) - 0.10) < 0.02
    assert abs(float(counts[1]) - 0.50) < 0.03
    assert abs(float(counts[2]) - 0.40) < 0.03


def test_marker_round_trip_and_tamper_detection():
    marker = _marker()
    assert validate_final_rescue_lineage_marker(marker) == marker
    tampered = dict(marker)
    tampered["sampler_probabilities"] = {"ordinary": 0.0}
    with pytest.raises(ValueError):
        validate_final_rescue_lineage_marker(tampered)


def test_marker_rejects_safety_failures_and_empty_failed_gate():
    with pytest.raises(ValueError):
        _marker(parent_failed_checks=["actual_soft_limits"])
    with pytest.raises(ValueError):
        _marker(failed_gate_failed_checks=[])
    with pytest.raises(ValueError):
        _marker(sampler_mix="v9")


def test_env_cfg_installs_only_command_samplers_and_drops_15000_stage():
    cfg = make_microban_teleop_v12_final_rescue_env_cfg(mix="v2")
    for name in ("twist", "foot_target", "hand_target"):
        assert cfg.commands[name].final_rescue_mix == "v2"
    stages = cfg.curriculum["staged_curriculum"].params["stages"]
    assert max(int(stage["step"]) for stage in stages) == 12_000 * 24
