"""CPU tests for the pose-release 14900->14999 final-scenario rescue."""

from __future__ import annotations

import importlib.util
import json
import math
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from mjlab_microban.scripts.evaluate_teleop_v12_tracking import (
    ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD,
    FINAL_COMPLETION_ALLOWANCE_PROFILE,
    FINAL_PROFILE,
    _acceptance,
    _scenarios,
    required_tracking_scenario_names,
)
from mjlab_microban.tasks.microban_teleop_v12_corner_rescue import (
    MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY,
    canonical_json_sha256,
    corner_rescue_marker,
    validate_corner_rescue_canonical_lineage,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
    MICROBAN_TELEOP_V12_RECIPE_REVISION,
)
from mjlab_microban.tasks.microban_teleop_v12_final_rescue import (
    MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY,
    MICROBAN_TELEOP_V12_FINAL_RESCUE_SCENARIOS,
    _FinalRescuePatternState,
    evaluator_scenario_commands,
    final_rescue_scenario_commands,
    validate_final_rescue_mix,
)
from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_final_rescue import (
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_MIXES,
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_TASK_ID,
    failed_final_gate_scenarios,
    final_gate_profile,
    hand_pose_release_final_rescue_marker,
    hand_pose_release_final_rescue_scenarios,
    is_hand_pose_release_final_rescue_marker,
    make_microban_teleop_v12_hand_pose_release_final_rescue_env_cfg,
    validate_hand_pose_release_final_rescue_marker,
)
from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_lineage import (
    HAND_POSE_RELEASE_LINEAGE_FRESH_CORNER_FINAL_RESCUE,
    HAND_POSE_RELEASE_LINEAGE_FRESH_FINAL_RESCUE,
    hand_pose_release_lineage,
)

SHA_PARENT = "a" * 64
SHA_FAILED = "c" * 64
SHA_REPORT = "d" * 64


def _stage_helpers():
    path = Path(__file__).with_name("test_teleop_v12_stage.py")
    spec = importlib.util.spec_from_file_location("_v12_stage_test_helpers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _corner() -> dict:
    return corner_rescue_marker(
        parent_checkpoint_sha256="e" * 64,
        parent_strict_tracking_report_sha256="f" * 64,
        hand_pose_release=True,
        parent_strict_failed_checks=("hand_tracking_rms",),
    )


def _marker(*, corner: dict | None = None, **overrides) -> dict:
    values = {
        "parent_checkpoint_sha256": SHA_PARENT,
        "failed_gate_checkpoint_sha256": SHA_FAILED,
        "failed_gate_tracking_report_sha256": SHA_REPORT,
        "failed_gate_failed_checks": ["actual_soft_limits", "twist_directional_response"],
        "failed_gate_failed_scenarios": ["bounded_both_feet", "mixed_forward_left"],
        "inherited_corner_rescue_marker_sha256": (
            None if corner is None else canonical_json_sha256(corner)
        ),
        "sampler_mix": "pr_v1",
    }
    values.update(overrides)
    return hand_pose_release_final_rescue_marker(**values)


def _infos(*, corner: dict | None, final: dict | None, recipe=None) -> dict:
    infos = {
        "microban_teleop_recipe_revision": (
            recipe or MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
        ),
        "env_state": {"common_step_counter": 15_000 * 24},
    }
    if corner is not None:
        infos[MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY] = corner
    if final is not None:
        infos[MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY] = final
    return infos


# --- mixes and replayed commands -------------------------------------------


@pytest.mark.parametrize("mix", sorted(MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_MIXES))
def test_mixes_keep_ordinary_commands_and_name_final_scenarios(mix):
    probabilities = MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_MIXES[mix]
    assert math.isclose(sum(probabilities.values()), 1.0)
    assert probabilities["ordinary"] >= 0.30
    scenarios = hand_pose_release_final_rescue_scenarios(mix)
    assert {"mixed_forward_left", "bounded_both_feet"} <= set(scenarios)
    assert set(scenarios) <= set(required_tracking_scenario_names(FINAL_PROFILE))


def test_registered_ordinary_shares():
    shares = {
        name: mix["ordinary"]
        for name, mix in MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_MIXES.items()
    }
    assert shares == {
        "pr_v1": 0.50,
        "pr_v2": 0.70,
        "pr_v3": 0.30,
        "pr_v4": 0.50,
        "pr_v5": 0.50,
        "pr_v6": 0.70,
    }
    # The two mix registries never overlap.
    for name in shares:
        with pytest.raises(ValueError):
            validate_final_rescue_mix(name)


def test_replayed_commands_mirror_the_final_pose_release_evaluator():
    names = ("mixed_forward_left", "bounded_both_feet", "mixed_backward_right")
    commands = evaluator_scenario_commands(names)
    evaluator = {item.name: item for item in _scenarios(final_gate_profile())}
    assert final_gate_profile() == FINAL_COMPLETION_ALLOWANCE_PROFILE
    for name in names:
        scenario = evaluator[name]
        assert commands[name]["twist"] == list(scenario.twist)
        assert commands[name]["foot_target"] == [list(v) for v in scenario.foot_target]
        assert commands[name]["hand_target"] == [list(v) for v in scenario.hand_target]
        assert commands[name]["hand_active"] == list(scenario.hand_active)
    assert commands["mixed_forward_left"]["hand_joint_pose_names"] == ["f", "b"]
    assert commands["bounded_both_feet"]["hand_active"] == [False, False]
    assert commands["bounded_both_feet"]["hand_joint_pose_names"] is None
    with pytest.raises(ValueError):
        evaluator_scenario_commands(("not_a_scenario",))


def test_canonical_final_rescue_commands_are_unchanged():
    commands = final_rescue_scenario_commands()
    assert tuple(commands) == MICROBAN_TELEOP_V12_FINAL_RESCUE_SCENARIOS
    assert commands["mixed_backward_right"]["hand_joint_pose_names"] == ["b", "f"]
    assert commands["max_keypoints_right"]["hand_joint_pose_names"] == ["B", "F"]
    assert all(item["hand_active"] == [True, True] for item in commands.values())


def test_pattern_state_draws_registered_pose_release_shares():
    torch.manual_seed(0)
    state = _FinalRescuePatternState(num_envs=8192, device="cpu", mix="pr_v1")
    assert state.scenarios == ("mixed_forward_left", "bounded_both_feet")
    counts = torch.bincount(state.pattern, minlength=3).double() / 8192
    for index, expected in enumerate((0.50, 0.30, 0.20)):
        assert abs(float(counts[index]) - expected) < 0.02
    with pytest.raises(ValueError):
        _FinalRescuePatternState(num_envs=4, device="cpu", mix="pr_v9")


def test_env_cfg_changes_only_the_command_samplers():
    from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release import (
        active_hand_arm_released_posture,
    )

    cfg = make_microban_teleop_v12_hand_pose_release_final_rescue_env_cfg(mix="pr_v2")
    for name in ("twist", "foot_target", "hand_target"):
        assert cfg.commands[name].final_rescue_mix == "pr_v2"
    assert cfg.rewards["pose"].func is active_hand_arm_released_posture
    stages = cfg.curriculum["staged_curriculum"].params["stages"]
    assert max(int(stage["step"]) for stage in stages) < 15_000 * 24
    play = make_microban_teleop_v12_hand_pose_release_final_rescue_env_cfg(play=True)
    assert not hasattr(play.commands["twist"], "final_rescue_mix")
    with pytest.raises(ValueError):
        make_microban_teleop_v12_hand_pose_release_final_rescue_env_cfg(mix="v1")


def test_task_is_registered():
    from mjlab.tasks.registry import list_tasks

    import mjlab_microban.tasks  # noqa: F401

    assert MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_TASK_ID in list_tasks()


def test_live_env_replays_scenarios_with_their_flags():
    from mjlab.envs import ManagerBasedRlEnv

    cfg = make_microban_teleop_v12_hand_pose_release_final_rescue_env_cfg(mix="pr_v3")
    cfg.scene.num_envs = 64
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        env.reset()
        commands = env.command_manager
        twist = commands.get_term("twist")
        foot = commands.get_term("foot_target")
        hand = commands.get_term("hand_target")
        state = twist._final_rescue
        assert state is foot._final_rescue is hand._final_rescue
        replay = evaluator_scenario_commands(state.scenarios)
        seen = set()
        for env_id in range(env.num_envs):
            pattern = int(state.pattern[env_id])
            if pattern == 0:
                continue
            name = state.scenarios[pattern - 1]
            seen.add(name)
            expected = replay[name]
            assert torch.allclose(
                twist.vel_command_b[env_id], torch.tensor(expected["twist"])
            )
            assert torch.allclose(
                foot.foot_target_offset_b[env_id],
                torch.tensor(expected["foot_target"]),
            )
            assert hand.is_active[env_id].tolist() == expected["hand_active"]
            both = name == "bounded_both_feet"
            assert bool(foot.is_both_feet_env[env_id]) is both
            assert bool(foot.is_single_support_env[env_id]) is (not both)
            if both:
                assert torch.count_nonzero(hand.hand_target_offset_b[env_id]) == 0
        assert seen == set(state.scenarios)
    finally:
        env.close()


# --- marker -----------------------------------------------------------------


def test_marker_round_trip_and_tamper_detection():
    marker = _marker()
    assert is_hand_pose_release_final_rescue_marker(marker)
    assert validate_hand_pose_release_final_rescue_marker(marker) == marker
    assert marker["parent_lineage"] == "fresh_chain"
    assert marker["failed_final_gate"]["tracking_profile"] == (
        FINAL_COMPLETION_ALLOWANCE_PROFILE
    )
    assert marker["source_recipe_revision"] == (
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
    )
    for key, value in (
        ("sampler_probabilities", {"ordinary": 0.0}),
        ("parent_lineage", "fresh_chain_model9900_corner_rescue"),
        ("gate_contract", "loosened"),
    ):
        tampered = dict(marker)
        tampered[key] = value
        with pytest.raises(ValueError):
            validate_hand_pose_release_final_rescue_marker(tampered)
    tampered = deepcopy(marker)
    tampered["failed_final_gate"]["tracking_profile"] = FINAL_PROFILE
    with pytest.raises(ValueError):
        validate_hand_pose_release_final_rescue_marker(tampered)


def test_marker_records_the_training_seed():
    assert _marker()["training"]["agent_seed"] == 42
    marker = _marker(training_seed=43)
    assert marker["training"]["environment_seed"] == marker["training"]["agent_seed"] == 43
    assert validate_hand_pose_release_final_rescue_marker(marker) == marker
    tampered = deepcopy(marker)
    tampered["training"]["environment_seed"] = 42
    with pytest.raises(ValueError):
        validate_hand_pose_release_final_rescue_marker(tampered)
    for seed in (-1, True, 2**31, "43"):
        with pytest.raises(ValueError):
            _marker(training_seed=seed)
    # The live environment count is recorded, not assumed.
    assert _marker()["training"]["num_envs"] == 2048
    small = _marker(num_envs=16)
    assert validate_hand_pose_release_final_rescue_marker(small)["training"]["num_envs"] == 16
    for count in (0, -1, True, 16.0):
        with pytest.raises(ValueError):
            _marker(num_envs=count)


def test_marker_requires_rescuable_failures_covered_by_the_mix():
    for overrides in (
        {"failed_gate_failed_checks": []},
        {"failed_gate_failed_checks": ["no_falls"]},
        {"failed_gate_failed_checks": ["target_column_ablation_response"]},
        {"failed_gate_failed_scenarios": []},
        {"failed_gate_failed_scenarios": ["mixed_forward_left", "bounded_both_feet"]},
        {"failed_gate_failed_scenarios": ["mixed_backward_right"]},
        {"sampler_mix": "v4"},
        {"failed_gate_checkpoint_sha256": SHA_PARENT},
    ):
        with pytest.raises(ValueError):
            _marker(**overrides)
    marker = _marker(
        failed_gate_failed_scenarios=["mixed_backward_right"], sampler_mix="pr_v4"
    )
    assert marker["sampler_scenarios"] == [
        "mixed_forward_left",
        "bounded_both_feet",
        "mixed_backward_right",
    ]


# --- lineage ----------------------------------------------------------------


def test_lineage_accepts_only_the_rescue_model14999():
    corner = _corner()
    marker = _marker(corner=corner)
    infos = _infos(corner=corner, final=marker)
    assert marker["parent_lineage"] == "fresh_chain_model9900_corner_rescue"
    assert (
        hand_pose_release_lineage(infos, iteration=14_999)
        == HAND_POSE_RELEASE_LINEAGE_FRESH_CORNER_FINAL_RESCUE
    )
    assert validate_corner_rescue_canonical_lineage(infos, iteration=14_999) == corner
    # Structural callers (HOME pose, packager recipe) pass no iteration.
    assert (
        hand_pose_release_lineage(infos, verify_parent=False)
        == HAND_POSE_RELEASE_LINEAGE_FRESH_CORNER_FINAL_RESCUE
    )
    for iteration in (14_900, 14_949, 14_998, 15_000):
        with pytest.raises(ValueError):
            hand_pose_release_lineage(infos, iteration=iteration)
    plain = _marker()
    assert (
        hand_pose_release_lineage(_infos(corner=None, final=plain), iteration=14_999)
        == HAND_POSE_RELEASE_LINEAGE_FRESH_FINAL_RESCUE
    )


def test_lineage_rejects_corner_mismatch_and_cross_recipe_markers():
    corner = _corner()
    with pytest.raises(ValueError):
        # Marker names no corner, checkpoint carries one.
        hand_pose_release_lineage(_infos(corner=corner, final=_marker()), iteration=14_999)
    with pytest.raises(ValueError):
        hand_pose_release_lineage(
            _infos(corner=None, final=_marker(corner=corner)), iteration=14_999
        )
    with pytest.raises(ValueError):
        validate_corner_rescue_canonical_lineage(
            _infos(corner=None, final=_marker(), recipe=MICROBAN_TELEOP_V12_RECIPE_REVISION),
            iteration=14_999,
        )
    from test_teleop_v12_final_rescue import _marker as canonical_marker

    with pytest.raises(ValueError):
        # The canonical final marker never joins the pose-release lineage.
        hand_pose_release_lineage(
            _infos(corner=None, final=canonical_marker()), iteration=14_999
        )


def test_home_pose_accepts_the_rescue_structurally():
    from mjlab_microban.tasks.microban_teleop_v12_home_pose import (
        TELEOP_V12_HOME_POSE_INFO_KEY,
        teleop_v12_home_pose_marker,
        validate_teleop_v12_home_pose,
    )

    corner = _corner()
    infos = _infos(corner=corner, final=_marker(corner=corner))
    infos[TELEOP_V12_HOME_POSE_INFO_KEY] = teleop_v12_home_pose_marker()
    validate_teleop_v12_home_pose(infos)
    infos[MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY] = dict(
        infos[MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY], sampler_mix="pr_v2"
    )
    with pytest.raises(ValueError):
        validate_teleop_v12_home_pose(infos)


# --- failed final gate report ----------------------------------------------


def _failed_final_report(*, soft=True, twist=True, fell=False):
    helpers = _stage_helpers()
    identity = {
        "path": "/runs/r/model_14999.pt",
        "sha256": SHA_FAILED,
        "iteration": 14_999,
        "completed_updates": 15_000,
    }
    report = helpers._tracking_report(identity, profile=FINAL_COMPLETION_ALLOWANCE_PROFILE)
    for item in report["results"]:
        if soft and item["name"] in ("bounded_both_feet", "mixed_forward_left"):
            item["maximum_actual_soft_limit_violation_rad"] = 0.0921
        if twist and item["name"] == "mixed_forward_left":
            axis = item["directional_response"]["vy_m_s"]
            axis.update(measured_mean=-0.0042, signed_response=-0.0042, passed=False)
            item["measured_velocity_body"]["vy_m_s"]["mean"] = -0.0042
            item["twist_directional_response_passed"] = False
        if fell and item["name"] == "max_hands_left":
            item["fell"] = True
    checks, status = _acceptance(report["results"], FINAL_COMPLETION_ALLOWANCE_PROFILE)
    report["checks"] = checks
    report["status"] = status
    return report


def test_gate_validator_still_refuses_soft_limit_and_twist_failures():
    from mjlab_microban.scripts.teleop_v12_stage import _validate_tracking_report

    identity = {"sha256": SHA_FAILED, "iteration": 14_999, "completed_updates": 15_000}
    report = _failed_final_report()
    assert report["status"] == "fail"
    assert 0.0921 > ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD
    with pytest.raises(ValueError):
        _validate_tracking_report(
            report, identity, recipe_revision=MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
        )
    for allowed in (
        frozenset(("actual_soft_limits",)),
        frozenset(("twist_directional_response",)),
        frozenset(("hand_tracking_rms", "foot_tracking_rms")),
    ):
        with pytest.raises(ValueError):
            _validate_tracking_report(
                report,
                identity,
                profile_override=FINAL_COMPLETION_ALLOWANCE_PROFILE,
                allowed_failed_checks=allowed,
            )
    assert (
        _validate_tracking_report(
            report,
            identity,
            profile_override=FINAL_COMPLETION_ALLOWANCE_PROFILE,
            allowed_failed_checks=frozenset(
                ("actual_soft_limits", "twist_directional_response")
            ),
        )
        == FINAL_COMPLETION_ALLOWANCE_PROFILE
    )
    # A passing report still validates as a gate.
    passing = _failed_final_report(soft=False, twist=False)
    assert passing["status"] == "pass"
    _validate_tracking_report(
        passing, identity, recipe_revision=MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
    )
    # A summary that disagrees with its per-axis evidence is refused.
    inconsistent = _failed_final_report(soft=False, twist=False)
    for item in inconsistent["results"]:
        if item["name"] == "mixed_forward_left":
            item["twist_directional_response_passed"] = False
    with pytest.raises(ValueError):
        _validate_tracking_report(
            inconsistent,
            identity,
            profile_override=FINAL_COMPLETION_ALLOWANCE_PROFILE,
            allowed_failed_checks=frozenset(("twist_directional_response",)),
        )


def test_failed_gate_report_validator(tmp_path):
    from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_final_rescue_runner import (
        validate_hand_pose_release_failed_final_gate_report,
    )

    report = _failed_final_report()
    assert failed_final_gate_scenarios(report) == ["bounded_both_feet", "mixed_forward_left"]
    path = tmp_path / "tracking.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    result = validate_hand_pose_release_failed_final_gate_report(path)
    assert result["failed_checks"] == ["actual_soft_limits", "twist_directional_response"]
    assert result["failed_scenarios"] == ["bounded_both_feet", "mixed_forward_left"]
    assert result["checkpoint_sha256"] == SHA_FAILED
    for bad in (
        _failed_final_report(soft=False, twist=False),  # nothing failed
        _failed_final_report(fell=True),  # non-rescuable
    ):
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError):
            validate_hand_pose_release_failed_final_gate_report(path)
    other_profile = _failed_final_report()
    other_profile["profile"] = "full_body_reachable_performance_perturbation_v2_deployed_accuracy_v1"
    path.write_text(json.dumps(other_profile), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_hand_pose_release_failed_final_gate_report(path)


def test_failed_gate_report_must_come_from_the_parent_run(tmp_path):
    from mjlab_microban.tasks.microban_teleop_v12_bootstrap import sha256_file
    from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_final_rescue_runner import (
        validate_hand_pose_release_failed_final_gate_report,
    )

    run = tmp_path / "run"
    other = tmp_path / "other"
    for directory in (run, other):
        directory.mkdir()
        (directory / "model_14900.pt").write_bytes(b"parent")
    failed = run / "model_14999.pt"
    failed.write_bytes(b"failed")
    report = _failed_final_report()
    report["checkpoint"]["path"] = str(failed)
    report["checkpoint"]["sha256"] = sha256_file(failed)
    path = tmp_path / "tracking.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    validate_hand_pose_release_failed_final_gate_report(
        path, parent_checkpoint=run / "model_14900.pt"
    )
    with pytest.raises(ValueError):
        validate_hand_pose_release_failed_final_gate_report(
            path, parent_checkpoint=other / "model_14900.pt"
        )
    failed.write_bytes(b"changed")
    with pytest.raises(ValueError):
        validate_hand_pose_release_failed_final_gate_report(
            path, parent_checkpoint=run / "model_14900.pt"
        )


# --- parent payload ----------------------------------------------------------


def _parent_payload(**info_overrides):
    from mjlab_microban.tasks.microban_teleop_v12_final_rescue import (
        MICROBAN_TELEOP_V12_FINAL_RESCUE_ACTIVE_COLUMNS,
    )

    infos = {
        "microban_teleop_training_contract_version": "12",
        "microban_teleop_recipe_revision": (
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
        ),
        "env_state": {"common_step_counter": 14_901 * 24},
        "active_actor_columns_at_save": list(
            MICROBAN_TELEOP_V12_FINAL_RESCUE_ACTIVE_COLUMNS
        ),
        MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY: _corner(),
    }
    infos.update(info_overrides)
    return {
        "iter": 14_900,
        "infos": infos,
        "optimizer_state_dict": {"state": {0: {"step": torch.tensor(298_020.0)}}},
    }


def test_parent_payload_contract():
    from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_final_rescue_runner import (
        validate_hand_pose_release_final_rescue_parent_payload,
    )

    assert validate_hand_pose_release_final_rescue_parent_payload(
        _parent_payload(), checkpoint_sha256=SHA_PARENT
    ) == _corner()
    assert (
        validate_hand_pose_release_final_rescue_parent_payload(
            _parent_payload(**{MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY: None}),
            checkpoint_sha256=SHA_PARENT,
        )
        is None
    )
    bad_payloads = [
        dict(_parent_payload(), iter=14_999),
        _parent_payload(microban_teleop_recipe_revision=MICROBAN_TELEOP_V12_RECIPE_REVISION),
        _parent_payload(env_state={"common_step_counter": 14_900 * 24}),
        _parent_payload(active_actor_columns_at_save=[]),
        _parent_payload(**{MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY: _marker(corner=_corner())}),
        dict(
            _parent_payload(),
            optimizer_state_dict={"state": {0: {"step": torch.tensor(298_000.0)}}},
        ),
    ]
    for payload in bad_payloads:
        with pytest.raises((ValueError, TypeError)):
            validate_hand_pose_release_final_rescue_parent_payload(
                payload, checkpoint_sha256=SHA_PARENT
            )


# --- packager resume ancestry crosses the staged seed -----------------------


def test_packager_ancestry_crosses_the_staged_seed(tmp_path):
    from mjlab_microban.scripts import export_teleop_v12_deployment as deployment

    def params(run: Path, parent: Path | None) -> None:
        (run / "params").mkdir(parents=True, exist_ok=True)
        text = (
            "resume: false\n"
            if parent is None
            else (
                "resume: true\n"
                f"load_run: ^{parent.parent.name}$\n"
                f"load_checkpoint: ^{parent.stem}[.]pt$\n"
            )
        )
        (run / "params" / "agent.yaml").write_text(text, encoding="utf-8")

    canary = tmp_path / "r_10000_to10100" / "model_10099.pt"
    segment = tmp_path / "r_10100_to15000" / "model_14900.pt"
    seed = tmp_path / "pr_final_rescue_seed_0123456789abcdef" / "model_14900.pt"
    final = tmp_path / "r_pr_final_rescue" / "model_14999.pt"
    for path in (canary, segment, seed, final):
        path.parent.mkdir(parents=True)
        path.write_bytes(path.parent.name.encode())
    params(canary.parent, None)
    params(segment.parent, canary)
    params(final.parent, seed)
    # Without the parent run's resume record the walk ends at the seed copy.
    assert deployment._resume_ancestry(final) == [seed.resolve()]
    (seed.parent / "params").mkdir()
    (seed.parent / "params" / "agent.yaml").write_bytes(
        (segment.parent / "params" / "agent.yaml").read_bytes()
    )
    assert deployment._resume_ancestry(final) == [seed.resolve(), canary.resolve()]


# --- the gate and the package record the rescue marker ----------------------


def _rescue_gate_fixture(root: Path, *, final: dict | None, corner: dict | None):
    """A pose-release model_14999 (optionally a final rescue) and its reports."""

    from mjlab_microban.tasks.mdp import MICROBAN_BILATERAL_SITE_ORDER_REVISION
    from mjlab_microban.tasks.microban_teleop_v12_actor import (
        TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION,
        teleop_v12_active_adapter_columns,
    )
    from mjlab_microban.tasks.microban_teleop_v12_bootstrap import sha256_file
    from mjlab_microban.tasks.microban_teleop_v12_home_pose import (
        TELEOP_V12_HOME_POSE_INFO_KEY,
        teleop_v12_home_pose_marker,
    )
    from mjlab_microban.tasks.microban_teleop_v12_lr_order import (
        BILATERAL_SITE_ORDER_INFO_KEY,
    )

    helpers = _stage_helpers()
    root.mkdir(parents=True, exist_ok=True)
    infos = {
        "microban_teleop_training_contract_version": "12",
        "microban_teleop_recipe_revision": (
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
        ),
        BILATERAL_SITE_ORDER_INFO_KEY: MICROBAN_BILATERAL_SITE_ORDER_REVISION,
        TELEOP_V12_HOME_POSE_INFO_KEY: teleop_v12_home_pose_marker(),
        "adapter_gradient_schedule_revision": (
            TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION
        ),
        "active_actor_columns_at_save": list(
            teleop_v12_active_adapter_columns(15_000 * 24)
        ),
        "env_state": {"common_step_counter": 15_000 * 24},
    }
    if corner is not None:
        infos[MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY] = corner
    if final is not None:
        infos[MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY] = final
    checkpoint = root / "model_14999.pt"
    torch.save({"iter": 14_999, "infos": infos}, checkpoint)
    identity = {
        "sha256": sha256_file(checkpoint),
        "iteration": 14_999,
        "completed_updates": 15_000,
    }
    onnx_path = root / "policy.onnx"
    onnx_path.write_bytes(b"unit-test-onnx")
    reports = {
        "locomotion_report": root / "locomotion.json",
        "tracking_report": root / "tracking.json",
        "onnx_report": root / "onnx.json",
    }
    reports["locomotion_report"].write_text(
        json.dumps(helpers._locomotion_report(identity))
    )
    reports["tracking_report"].write_text(
        json.dumps(
            helpers._tracking_report(identity, profile=FINAL_COMPLETION_ALLOWANCE_PROFILE)
        )
    )
    reports["onnx_report"].write_text(
        json.dumps(helpers._onnx_report(identity, onnx_path))
    )
    return checkpoint, reports, infos


@pytest.mark.parametrize("with_corner", [False, True])
def test_final_gate_records_and_requires_the_rescue_marker(tmp_path, with_corner):
    from mjlab_microban.scripts.teleop_v12_stage import create_gate, validate_gate

    corner = _corner() if with_corner else None
    marker = _marker(corner=corner, failed_gate_failed_scenarios=["mixed_forward_left"])
    checkpoint, reports, _ = _rescue_gate_fixture(
        tmp_path / "rescue", final=marker, corner=corner
    )
    gate = create_gate(checkpoint=checkpoint, **reports)
    assert gate["tracking_profile"] == FINAL_COMPLETION_ALLOWANCE_PROFILE
    assert gate[MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY] == marker
    assert gate.get(MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY) == corner
    gate_path = tmp_path / "rescue_gate.json"
    gate_path.write_text(json.dumps(gate))
    assert validate_gate(gate_path, checkpoint) == gate

    missing = deepcopy(gate)
    del missing[MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY]
    other = deepcopy(gate)
    other[MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY] = _marker(
        corner=corner, sampler_mix="pr_v3", failed_gate_failed_scenarios=["mixed_forward_left"]
    )
    reseeded = deepcopy(gate)
    reseeded[MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY] = _marker(
        corner=corner, training_seed=43, failed_gate_failed_scenarios=["mixed_forward_left"]
    )
    for tampered in (missing, other, reseeded):
        gate_path.write_text(json.dumps(tampered))
        with pytest.raises(ValueError, match="identity mismatch"):
            validate_gate(gate_path, checkpoint)

    # An ordinary pose-release final records no marker and refuses one.
    plain_checkpoint, plain_reports, _ = _rescue_gate_fixture(
        tmp_path / "plain", final=None, corner=corner
    )
    plain = create_gate(checkpoint=plain_checkpoint, **plain_reports)
    assert MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY not in plain
    plain_path = tmp_path / "plain_gate.json"
    plain_path.write_text(json.dumps(plain))
    assert validate_gate(plain_path, plain_checkpoint) == plain
    plain_path.write_text(
        json.dumps({**plain, MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY: marker})
    )
    with pytest.raises(ValueError):
        validate_gate(plain_path, plain_checkpoint)


def test_package_metadata_names_the_rescue_marker():
    from mjlab_microban.scripts.export_teleop_v12_deployment import (
        _final_rescue_metadata,
    )

    corner = _corner()
    marker = _marker(corner=corner)
    infos = _infos(corner=corner, final=marker)
    gate = {MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY: deepcopy(marker)}
    metadata = _final_rescue_metadata(gate, infos)
    assert set(metadata) == {
        "v12_final_rescue_marker_revision",
        "v12_final_rescue_marker_json",
        "v12_final_rescue_marker_sha256",
        "v12_final_rescue_training_replay",
        "v12_final_rescue_final_gate_held_out",
    }
    assert json.loads(metadata["v12_final_rescue_marker_json"]) == marker
    assert metadata["v12_final_rescue_marker_sha256"] == canonical_json_sha256(marker)
    assert metadata["v12_final_rescue_marker_revision"] == marker["revision"]
    # The package says the final gate's scenarios were trained on.
    assert metadata["v12_final_rescue_final_gate_held_out"] == "false"
    assert metadata["v12_final_rescue_training_replay"] == "evaluator_scenario_commands"
    pushed = _marker(corner=corner, sampler_mix="pr_v5")
    pushed_metadata = _final_rescue_metadata(
        {MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY: deepcopy(pushed)},
        _infos(corner=corner, final=pushed),
    )
    assert pushed_metadata["v12_final_rescue_training_replay"] == (
        "evaluator_scenario_commands_and_perturbation"
    )
    assert pushed_metadata["v12_final_rescue_final_gate_held_out"] == "false"
    # Every value is a string (ONNX metadata_props).
    assert all(isinstance(value, str) for value in metadata.values())
    for bad_gate in (
        {},
        {MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY: _marker(corner=corner, sampler_mix="pr_v3")},
        {MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY: _marker(corner=corner, training_seed=43)},
    ):
        with pytest.raises(ValueError):
            _final_rescue_metadata(bad_gate, infos)
    # A tampered checkpoint marker is refused before the comparison.
    tampered = deepcopy(marker)
    tampered["sampler_probabilities"]["ordinary"] = 0.9
    with pytest.raises(ValueError):
        _final_rescue_metadata(
            {MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY: tampered},
            _infos(corner=corner, final=tampered),
        )
    # An ordinary final ships no rescue fields and refuses a gate naming one.
    plain = _infos(corner=corner, final=None)
    assert _final_rescue_metadata({}, plain) == {}
    with pytest.raises(ValueError):
        _final_rescue_metadata(gate, plain)


# --- a load re-checks that the failed report is from the parent's run --------


def test_runner_load_resolves_the_staged_parent_run(tmp_path):
    from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_final_rescue_runner import (
        staged_hand_pose_release_final_rescue_source,
    )

    run = tmp_path / "lean_v12_pr_10100_to15000"
    seed = tmp_path / "pr_final_rescue_seed_0123456789abcdef"
    for directory in (run, seed):
        directory.mkdir()
        (directory / "model_14900.pt").write_bytes(b"parent")
    staged = seed / "model_14900.pt"
    record = seed / "parent_run.txt"
    # A hand-staged seed without the launcher's record is refused.
    with pytest.raises(ValueError, match="parent run"):
        staged_hand_pose_release_final_rescue_source(staged)
    record.write_text(f"{run.name}\n", encoding="utf-8")
    assert staged_hand_pose_release_final_rescue_source(staged) == (
        run / "model_14900.pt"
    ).resolve()
    for bad in (f"{seed.name}\n", "../x\n", "a\nb\n", "missing_run\n", ""):
        record.write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError):
            staged_hand_pose_release_final_rescue_source(staged)
    record.write_text(f"{run.name}\n", encoding="utf-8")
    (run / "model_14900.pt").write_bytes(b"different")
    with pytest.raises(ValueError):
        staged_hand_pose_release_final_rescue_source(staged)


# --- wiring: the packager entry point writes the marker fields ---------------


def test_build_deployment_metadata_writes_the_rescue_marker(tmp_path):
    from test_teleop_v12_deployment import (
        _bootstrap,
        _evidence,
        _microban_identity,
        _sha,
    )

    from mjlab_microban.scripts import export_teleop_v12_deployment as deployment

    corner = _corner()
    marker = _marker(corner=corner)
    checkpoint = tmp_path / "model_14999.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_path = tmp_path / "gate.json"
    gate_path.write_text("{}", encoding="utf-8")
    gate, locomotion, tracking, onnx_report, infos = _evidence(tmp_path)
    infos["microban_teleop_recipe_revision"] = (
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
    )
    infos[MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY] = corner
    gate["checkpoint_sha256"] = _sha(checkpoint)
    gate["tracking_profile"] = FINAL_COMPLETION_ALLOWANCE_PROFILE

    def build(gate_value: dict, infos_value: dict) -> dict:
        return deployment.build_v12_deployment_metadata(
            checkpoint=checkpoint,
            checkpoint_sha256=_sha(checkpoint),
            gate_path=gate_path,
            gate=gate_value,
            infos=infos_value,
            bootstrap=_bootstrap(),
            locomotion=locomotion,
            tracking=tracking,
            onnx_report=onnx_report,
            packager_parity={
                "reference_maximum_absolute_error": 1.0e-6,
                "onnxruntime_cpu_maximum_absolute_error": 2.0e-6,
            },
            microban_source_identity=_microban_identity(),
        )

    keys = {
        "v12_final_rescue_marker_revision",
        "v12_final_rescue_marker_json",
        "v12_final_rescue_marker_sha256",
        "v12_final_rescue_training_replay",
        "v12_final_rescue_final_gate_held_out",
    }
    # An ordinary pose-release final ships none of the rescue fields.
    plain = build(deepcopy(gate), deepcopy(infos))
    assert not keys.intersection(plain)
    assert plain["v12_tracking_profile"] == FINAL_COMPLETION_ALLOWANCE_PROFILE

    rescue_infos = {**deepcopy(infos), MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY: marker}
    rescue_gate = {**deepcopy(gate), MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY: marker}
    metadata = build(rescue_gate, rescue_infos)
    assert keys <= set(metadata)
    assert json.loads(metadata["v12_final_rescue_marker_json"]) == marker
    assert metadata["v12_final_rescue_marker_sha256"] == canonical_json_sha256(marker)
    assert metadata["v12_final_rescue_marker_revision"] == marker["revision"]
    assert {
        key: value for key, value in metadata.items() if key not in keys
    } == plain
    for bad_gate, bad_infos in (
        (deepcopy(gate), rescue_infos),  # gate omits the checkpoint's marker
        (
            {
                **deepcopy(gate),
                MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY: _marker(
                    corner=corner, sampler_mix="pr_v3"
                ),
            },
            rescue_infos,
        ),
        (rescue_gate, deepcopy(infos)),  # gate names a rescue the final lacks
    ):
        with pytest.raises(ValueError, match="final rescue"):
            build(bad_gate, bad_infos)


# --- wiring: every runner load re-checks the staged parent's own run ---------


class _ReachedPoseReleaseLoad(Exception):
    pass


def test_runner_load_refuses_a_hand_made_seed_folder(tmp_path, monkeypatch):
    from mjlab_microban.tasks import (
        microban_teleop_v12_hand_pose_release_final_rescue_runner as runner_module,
    )
    from mjlab_microban.tasks.microban_teleop_v12_bootstrap import sha256_file
    from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_final_rescue import (
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_FAILED_GATE_REPORT_FILENAME,
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_PARENT_RUN_FILENAME,
    )

    def reached(self, *args, **kwargs):
        raise _ReachedPoseReleaseLoad

    monkeypatch.setattr(
        runner_module.MicrobanTeleopV12HandPoseReleaseOnPolicyRunner, "load", reached
    )
    load = (
        runner_module.MicrobanTeleopV12HandPoseReleaseFinalRescueOnPolicyRunner.load
    )

    run = tmp_path / "lean_v12_pr_10100_to15000"
    other = tmp_path / "lean_v12_pr_10100_to15000_other"
    for directory in (run, other):
        directory.mkdir()
        torch.save(_parent_payload(), directory / "model_14900.pt")
        (directory / "model_14999.pt").write_bytes(directory.name.encode())

    def stage(name: str, *, failed_run: Path, record: str | None) -> Path:
        seed = tmp_path / name
        seed.mkdir()
        staged = seed / "model_14900.pt"
        staged.write_bytes((run / "model_14900.pt").read_bytes())
        failed = failed_run / "model_14999.pt"
        report = _failed_final_report()
        report["checkpoint"]["path"] = str(failed)
        report["checkpoint"]["sha256"] = sha256_file(failed)
        (
            seed
            / MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_FAILED_GATE_REPORT_FILENAME
        ).write_text(json.dumps(report), encoding="utf-8")
        if record is not None:
            (
                seed / MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_PARENT_RUN_FILENAME
            ).write_text(f"{record}\n", encoding="utf-8")
        return staged

    # The launcher's staging reaches the pose-release load.
    good = stage("pr_final_rescue_seed_good", failed_run=run, record=run.name)
    with pytest.raises(_ReachedPoseReleaseLoad):
        load(object(), str(good))
    # A hand-made seed without the parent-run record never reaches it, even
    # though its failed report and model_14900 are internally consistent.
    unrecorded = stage("pr_final_rescue_seed_unrecorded", failed_run=run, record=None)
    with pytest.raises(ValueError, match="parent run"):
        load(object(), str(unrecorded))
    # A failed report from another run than the recorded parent run.
    crossed = stage("pr_final_rescue_seed_crossed", failed_run=other, record=run.name)
    with pytest.raises(ValueError, match="not from the parent's run"):
        load(object(), str(crossed))
    # A recorded parent run whose model_14900 is not the staged copy.
    torch.save(
        _parent_payload(**{MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY: None}),
        other / "model_14900.pt",
    )
    swapped = stage("pr_final_rescue_seed_swapped", failed_run=other, record=other.name)
    with pytest.raises(ValueError, match="differs from its run"):
        load(object(), str(swapped))


# --- lineage and parent refusals pinned to their own checks ------------------


def _pinned_release_switch_marker(monkeypatch) -> dict:
    from mjlab_microban.tasks import (
        microban_teleop_v12_hand_pose_release_lineage as lineage_module,
    )

    monkeypatch.setattr(
        lineage_module,
        "HAND_POSE_RELEASE_RECIPE_SWITCH_PARENT_CHECKPOINT_SHA256",
        "d" * 64,
    )
    return lineage_module.hand_pose_release_recipe_switch_marker(
        parent_checkpoint_path="repo://logs/rsl_rl/v12/run_7000_to7100/model_7099.pt",
        parent_checkpoint_sha256="d" * 64,
        parent_stage_gate_path="repo://artifacts/teleop_v12_gates/run_model_7099_gate.json",
        parent_stage_gate_sha256="b" * 64,
    )


_EXPERIMENTAL_SWITCH = {
    "schema_version": 1,
    "release_eligible": False,
    "parent_recipe_revision": MICROBAN_TELEOP_V12_RECIPE_REVISION,
    "recipe_revision": MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
    "parent_checkpoint_sha256": "0" * 64,
    "parent_iteration": 7099,
}


def test_lineage_refuses_a_final_rescue_on_a_recipe_switch(monkeypatch):
    from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_lineage import (
        HAND_POSE_RELEASE_LINEAGE_EXPERIMENTAL_SWITCH,
        HAND_POSE_RELEASE_LINEAGE_RELEASE_SWITCH,
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_EXPERIMENTAL_SWITCH_INFO_KEY,
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_SWITCH_INFO_KEY,
    )

    release = _pinned_release_switch_marker(monkeypatch)
    release_infos = {
        **_infos(corner=None, final=None),
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_SWITCH_INFO_KEY: release,
    }
    experimental_infos = {
        **_infos(corner=None, final=None),
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_EXPERIMENTAL_SWITCH_INFO_KEY: deepcopy(
            _EXPERIMENTAL_SWITCH
        ),
    }
    # The switch lineages themselves classify as before.
    assert (
        hand_pose_release_lineage(release_infos, iteration=14_999, verify_parent=False)
        == HAND_POSE_RELEASE_LINEAGE_RELEASE_SWITCH
    )
    assert (
        hand_pose_release_lineage(
            experimental_infos,
            iteration=14_999,
            allow_experimental=True,
            verify_parent=False,
        )
        == HAND_POSE_RELEASE_LINEAGE_EXPERIMENTAL_SWITCH
    )
    # A forged model_14999 adding the (otherwise valid) final-rescue marker.
    for infos, allow_experimental in (
        (release_infos, False),
        (experimental_infos, True),
    ):
        forged = {**infos, MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY: _marker()}
        for kwargs in (
            {"iteration": 14_999, "verify_parent": False},
            {"verify_parent": False},
        ):
            with pytest.raises(ValueError, match="apply only to a fresh"):
                hand_pose_release_lineage(
                    forged, allow_experimental=allow_experimental, **kwargs
                )


def test_parent_payload_refusals_are_pinned_to_their_checks(monkeypatch):
    from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_final_rescue_runner import (
        validate_hand_pose_release_final_rescue_parent_payload,
    )
    from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_lineage import (
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_SWITCH_INFO_KEY,
    )

    with pytest.raises(ValueError, match="already carries a final rescue"):
        validate_hand_pose_release_final_rescue_parent_payload(
            _parent_payload(
                **{MICROBAN_TELEOP_V12_FINAL_RESCUE_INFO_KEY: _marker(corner=_corner())}
            ),
            checkpoint_sha256=SHA_PARENT,
        )
    # A release-eligible recipe-switch model_14900 is a valid pose-release
    # checkpoint, but not a fresh chain: the rescue refuses it.
    release = _pinned_release_switch_marker(monkeypatch)
    with pytest.raises(ValueError, match="must be a fresh pose-release chain"):
        validate_hand_pose_release_final_rescue_parent_payload(
            _parent_payload(
                **{
                    MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY: None,
                    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_SWITCH_INFO_KEY: release,
                }
            ),
            checkpoint_sha256=SHA_PARENT,
        )


# --- evaluator perturbation replay (pr_v5 / pr_v6) ------------------------------


def test_scenario_push_mixes_repeat_the_pr_v1_and_pr_v2_shares():
    from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_final_rescue import (
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_SCENARIO_PUSH_MIXES,
        hand_pose_release_final_rescue_uses_scenario_push,
    )

    mixes = MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_MIXES
    assert MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_SCENARIO_PUSH_MIXES == {
        "pr_v5",
        "pr_v6",
    }
    assert mixes["pr_v5"] == mixes["pr_v1"]
    assert mixes["pr_v6"] == mixes["pr_v2"]
    assert [m for m in sorted(mixes) if hand_pose_release_final_rescue_uses_scenario_push(m)] == [
        "pr_v5",
        "pr_v6",
    ]


def test_scenario_push_is_the_final_evaluator_perturbation():
    from mjlab_microban.scripts.evaluate_teleop_v12_tracking import (
        _tracking_cfg,
        tracking_profile_uses_perturbation,
    )
    from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_final_rescue import (
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_SCENARIO_PUSH,
    )

    assert tracking_profile_uses_perturbation(final_gate_profile())
    evaluator = _tracking_cfg(seed=42, steps=300, perturbation=True).events["push_robot"]
    push = MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_FINAL_RESCUE_SCENARIO_PUSH
    assert tuple(evaluator.interval_range_s) == push["interval_range_s"]
    assert evaluator.params["velocity_range"] == push["velocity_range"]
    assert set(evaluator.params) == {"velocity_range"}
    rescue = make_microban_teleop_v12_hand_pose_release_final_rescue_env_cfg(mix="pr_v6")
    replay = rescue.events["final_rescue_scenario_push"]
    for key in ("mode", "is_global_time", "min_step_count_between_reset"):
        assert getattr(replay, key) == getattr(evaluator, key)


def test_scenario_push_env_cfg_splits_the_push_by_episode_kind():
    from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release import (
        make_microban_teleop_v12_hand_pose_release_env_cfg,
    )
    from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_final_rescue import (
        push_ordinary_episodes,
        push_replayed_episodes,
    )

    base = make_microban_teleop_v12_hand_pose_release_env_cfg(play=False)
    for mix in ("pr_v1", "pr_v2", "pr_v3", "pr_v4"):
        cfg = make_microban_teleop_v12_hand_pose_release_final_rescue_env_cfg(mix=mix)
        assert set(cfg.events) == set(base.events)
        assert cfg.events["push_robot"].func is base.events["push_robot"].func
        assert cfg.events["push_robot"].params == base.events["push_robot"].params
    cfg = make_microban_teleop_v12_hand_pose_release_final_rescue_env_cfg(mix="pr_v5")
    assert set(cfg.events) == set(base.events) | {"final_rescue_scenario_push"}
    ordinary = cfg.events["push_robot"]
    assert ordinary.func is push_ordinary_episodes
    assert ordinary.interval_range_s == base.events["push_robot"].interval_range_s
    assert ordinary.params == {
        "velocity_range": base.events["push_robot"].params["velocity_range"],
        "final_rescue_mix": "pr_v5",
    }
    replay = cfg.events["final_rescue_scenario_push"]
    assert replay.func is push_replayed_episodes
    assert replay.interval_range_s == (1.0, 1.0)
    assert replay.params == {
        "velocity_range": {"x": (0.35, 0.35), "y": (-0.20, -0.20)},
        "final_rescue_mix": "pr_v5",
    }
    # The staged curriculum keeps driving the ordinary push range.
    from mjlab_microban.tasks.microban_teleop_env_cfg import _set_push_velocity_range

    class _Env:
        class event_manager:
            @staticmethod
            def get_term_cfg(name):
                return cfg.events[name]

    _set_push_velocity_range(_Env, x=(-0.5, 0.5), y=(-0.5, 0.5))
    assert ordinary.params["velocity_range"] == {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}
    assert ordinary.params["final_rescue_mix"] == "pr_v5"


def test_scenario_push_terms_select_episodes_by_pattern(monkeypatch):
    import mjlab.envs.mdp as envs_mdp

    from mjlab_microban.tasks.microban_teleop_v12_final_rescue import (
        _PATTERN_STATE_ATTRIBUTE,
    )
    from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_final_rescue import (
        push_ordinary_episodes,
        push_replayed_episodes,
    )

    calls = []
    monkeypatch.setattr(
        envs_mdp,
        "push_by_setting_velocity",
        lambda env, ids, velocity_range: calls.append((ids.tolist(), velocity_range)),
    )

    class _Env:
        num_envs = 6
        device = "cpu"

    env = _Env()
    state = _FinalRescuePatternState(num_envs=6, device="cpu", mix="pr_v6")
    state.pattern = torch.tensor([0, 1, 0, 2, 1, 0])
    setattr(env, _PATTERN_STATE_ATTRIBUTE, state)
    ids = torch.tensor([0, 1, 2, 3, 4, 5])
    push_ordinary_episodes(env, ids, {"x": (-0.5, 0.5)}, "pr_v6")
    push_replayed_episodes(env, ids, {"x": (0.35, 0.35)}, "pr_v6")
    push_replayed_episodes(env, torch.tensor([0, 2]), {"x": (0.35, 0.35)}, "pr_v6")
    push_ordinary_episodes(env, None, {"x": (-0.5, 0.5)}, "pr_v6")
    assert calls == [
        ([0, 2, 5], {"x": (-0.5, 0.5)}),
        ([1, 3, 4], {"x": (0.35, 0.35)}),
        ([0, 2, 5], {"x": (-0.5, 0.5)}),
    ]
    with pytest.raises(ValueError):
        push_ordinary_episodes(env, ids, {"x": (-0.5, 0.5)}, "pr_v1")


def test_scenario_push_marker_records_the_push_only_for_its_mixes():
    plain = _marker(sampler_mix="pr_v1")
    assert "scenario_push" not in plain
    pushed = _marker(sampler_mix="pr_v5")
    assert validate_hand_pose_release_final_rescue_marker(pushed) == pushed
    assert pushed["scenario_push"] == {
        "source": "final_profile_tracking_evaluator_perturbation",
        "scope": "replayed_episodes_only",
        "interval_range_s": [1.0, 1.0],
        "velocity_range": {"x": [0.35, 0.35], "y": [-0.20, -0.20]},
        "ordinary_push": "staged_curriculum_random_push_on_ordinary_episodes_only",
    }
    without = dict(pushed)
    del without["scenario_push"]
    with pytest.raises(ValueError):
        validate_hand_pose_release_final_rescue_marker(without)
    grafted = dict(plain)
    grafted["scenario_push"] = pushed["scenario_push"]
    with pytest.raises(ValueError):
        validate_hand_pose_release_final_rescue_marker(grafted)
    # Same mix shares otherwise: only the mix name and the push record differ.
    assert {k for k in pushed if pushed.get(k) != plain.get(k)} == {
        "sampler_mix",
        "scenario_push",
    }


def test_live_env_pushes_replayed_episodes_with_the_evaluator_kick():
    from mjlab.envs import ManagerBasedRlEnv

    cfg = make_microban_teleop_v12_hand_pose_release_final_rescue_env_cfg(mix="pr_v5")
    cfg.scene.num_envs = 32
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        env.reset()
        state = env.command_manager.get_term("twist")._final_rescue
        names = env.event_manager.active_terms["interval"]
        assert "push_robot" in names and "final_rescue_scenario_push" in names
        robot = env.scene["robot"]
        before = robot.data.root_link_vel_w.clone()
        ids = torch.arange(env.num_envs)
        replayed = ids[state.pattern != 0]
        ordinary = ids[state.pattern == 0]
        assert len(replayed) > 0 and len(ordinary) > 0
        term = env.event_manager.get_term_cfg("final_rescue_scenario_push")
        term.func(env, ids, **term.params)
        env.sim.forward()
        after = robot.data.root_link_vel_w
        delta = after - before
        assert torch.allclose(delta[replayed, 0], torch.full((len(replayed),), 0.35), atol=1e-5)
        assert torch.allclose(delta[replayed, 1], torch.full((len(replayed),), -0.20), atol=1e-5)
        assert torch.count_nonzero(delta[ordinary]) == 0
    finally:
        env.close()
