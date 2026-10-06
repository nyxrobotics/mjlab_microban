from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from home_cases import CENTERED_HOME_TAG, FORWARD_LEAN_HOME_TAG, home_tag  # noqa: E402

from mjlab_microban.scripts import export_teleop_v12_deployment as deployment
from mjlab_microban.policy_contract import POLICY_CONTRACT
from mjlab_microban.robot.microban_constants import SERVO_KP_POLICY
from mjlab_microban.schedules import (
    PICO_MIN_FINAL_UPDATES,
    PICO_TOTAL_UPDATES,
    pico_schedule_record,
)
from mjlab_microban.tasks.mdp import MICROBAN_BILATERAL_SITE_ORDER_REVISION
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
)
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    LEGACY_TO_TELEOP_OBSERVATION_INDEX,
    LEGACY_VELOCITY_ACTOR_STATE_KEYS,
    LEGACY_VELOCITY_NORMALIZER_EPS,
    TELEOP_V12_ACTOR_TOPOLOGY,
    TELEOP_V12_BOOTSTRAP_MAPPING_VERSION,
    TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
)
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import (
    LegacyTeleopProbeIdentity,
    LegacyVelocitySourceIdentity,
    TeleopV12BootstrapProvenance,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_ACTION_CLIP,
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
)
from mjlab_microban.tasks.microban_teleop_v12_home_pose import (
    TELEOP_V12_HOME_POSE_INFO_KEY,
    teleop_v12_home_pose_marker,
)
from mjlab_microban.tasks.microban_teleop_v12_runner import (
    BILATERAL_SITE_ORDER_INFO_KEY,
    require_bilateral_site_order,
)
from mjlab_microban.teleop_v12_safety import (
    ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_DEG,
    ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD,
    COMMANDED_TARGET_SOFT_LIMIT_EXCESS_MAX_RAD,
)

# Synthetic identities: the velocity source is chosen per chain.
LEGACY_VELOCITY_CHECKPOINT_SHA256 = "1" * 64
PINNED_LEGACY_TELEOP_PROBE_SHA256 = "2" * 64


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap() -> TeleopV12BootstrapProvenance:
    return TeleopV12BootstrapProvenance(
        schema_version=2,
        source=LegacyVelocitySourceIdentity(
            path="/runs/velocity/model_14999.pt",
            sha256=LEGACY_VELOCITY_CHECKPOINT_SHA256,
            iteration=14_999,
            normalizer_count=1_474_560_000,
        ),
        probe=LegacyTeleopProbeIdentity(
            path="repo://artifacts/legacy_velocity_teleop_probe.json",
            sha256=PINNED_LEGACY_TELEOP_PROBE_SHA256,
            scenario_count=9,
            steps=300,
            settle_steps=50,
            seed=42,
        ),
        mapping_version=TELEOP_V12_BOOTSTRAP_MAPPING_VERSION,
        source_to_target_columns=LEGACY_TO_TELEOP_OBSERVATION_INDEX,
        new_trainable_columns=TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
        target_actor_topology=TELEOP_V12_ACTOR_TOPOLOGY,
        normalizer_eps=LEGACY_VELOCITY_NORMALIZER_EPS,
        previous_action_semantics="raw_actor_output",
        action_clip=list(MICROBAN_TELEOP_V12_ACTION_CLIP),
        frozen_tensors=tuple(
            sorted(LEGACY_VELOCITY_ACTOR_STATE_KEYS - {"mlp.0.weight"})
        ),
    )


def _evidence(root: Path) -> tuple[dict, dict, dict, dict, dict]:
    checkpoint_sha = "1" * 64
    reports = {
        "locomotion": str(root / "locomotion.json"),
        "tracking": str(root / "tracking.json"),
        "onnx": str(root / "onnx.json"),
    }
    gate = {
        "schema_version": 3,
        "gate": "microban_teleop_v12_stage",
        "status": "pass",
        TELEOP_V12_HOME_POSE_INFO_KEY: teleop_v12_home_pose_marker(),
        "checkpoint_sha256": checkpoint_sha,
        "iteration": PICO_TOTAL_UPDATES - 1,
        "completed_updates": PICO_TOTAL_UPDATES,
        "tracking_profile": deployment.FINAL_PROFILE,
        "reports": reports,
        "report_sha256": {
            "locomotion": "2" * 64,
            "tracking": "3" * 64,
            "onnx": "4" * 64,
        },
        "onnx": {"path": str(root / "gate.onnx"), "sha256": "5" * 64},
    }
    locomotion = {
        "gate": "microban_teleop_v12_neutral_locomotion_9x300",
        "status": "pass",
        "settings": {"seed": 42, "steps": 300, "settle_steps": 50},
        "summary": {
            "scenario_count": 9,
            "fall_scenario_count": 0,
            "nonfinite_scenario_count": 0,
            "directionally_correct_scenario_count": 8,
            "directional_scenario_count": 8,
        },
        "results": [{"maximum_actual_soft_limit_violation_rad": 0.0} for _ in range(9)],
    }

    def summary(minimum: float, maximum: float) -> dict[str, list[float]]:
        return {
            "minimum": [minimum] * 18,
            "maximum": [maximum] * 18,
            "absolute_maximum": [max(abs(minimum), abs(maximum))] * 18,
        }

    tracking = {
        "raw_action_envelope": {
            "joint_names": list(MICROBAN_TELEOP_ACTION_JOINT_NAMES),
            "v12": summary(-3.0, 4.0),
            "legacy_source": summary(-2.0, 2.0),
            "learned_minus_source": summary(-1.0, 1.0),
            "scenario_count": 12,
            "step_count": 3_600,
        },
        "runtime_smoke_observations": [
            [0.0] * 5 + [-1.0] + [0.001 * row] * 77 for row in range(16)
        ],
    }
    onnx_report = {
        "gate": "microban_teleop_v12_checkpoint_onnx",
        "neutral_legacy_parity": {
            "samples": 10_000,
            "maximum_absolute_error": 1.0e-6,
        },
        "onnx": {
            "reference_samples": 64,
            "tolerance": 2.0e-5,
            "reference_evaluator_maximum_absolute_error": 2.0e-6,
            "onnxruntime_cpu_maximum_absolute_error": 3.0e-6,
        },
    }
    infos = {
        "microban_teleop_recipe_revision": MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
        TELEOP_V12_HOME_POSE_INFO_KEY: teleop_v12_home_pose_marker(),
        "trainable_actor_parameters": ["mlp.0.weight"],
        "trainable_actor_columns": list(TELEOP_V12_EXTRA_OBSERVATION_COLUMNS),
        "active_actor_columns_at_save": list(TELEOP_V12_EXTRA_OBSERVATION_COLUMNS),
        deployment.TELEOP_V12_BOOTSTRAP_INFO_KEY: {},
        BILATERAL_SITE_ORDER_INFO_KEY: MICROBAN_BILATERAL_SITE_ORDER_REVISION,
    }
    return gate, locomotion, tracking, onnx_report, infos


FINAL_NAME = f"model_{PICO_TOTAL_UPDATES - 1}.pt"


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("status", "fail"),
        ("schema_version", 2),
        ("iteration", PICO_TOTAL_UPDATES - 2),
        ("completed_updates", PICO_MIN_FINAL_UPDATES - 1),
        ("completed_updates", PICO_TOTAL_UPDATES + 1),
        ("tracking_profile", "expanded_locomotion_tracking_v1"),
    ],
)
def test_final_gate_rejects_every_other_identity(
    tmp_path: Path, field: str, bad_value: object
) -> None:
    checkpoint = tmp_path / FINAL_NAME
    gate, *_ = _evidence(tmp_path)
    gate[field] = bad_value
    with pytest.raises(ValueError, match="requires the passing gate"):
        deployment._require_final_gate(
            gate,
            checkpoint=checkpoint,
            checkpoint_sha256="1" * 64,
        )


def test_an_adopted_earlier_checkpoint_is_packaged(tmp_path: Path) -> None:
    gate, *_ = _evidence(tmp_path)
    gate.update(iteration=PICO_MIN_FINAL_UPDATES - 1, completed_updates=PICO_MIN_FINAL_UPDATES)
    deployment._require_final_gate(
        gate, checkpoint=tmp_path / f"model_{PICO_MIN_FINAL_UPDATES - 1}.pt", checkpoint_sha256="1" * 64
    )
    with pytest.raises(ValueError, match="model_<iteration>"):
        deployment._require_final_gate(
            gate, checkpoint=tmp_path / FINAL_NAME, checkpoint_sha256="1" * 64
        )


def test_final_gate_profile_is_the_one_final_profile(tmp_path: Path) -> None:
    checkpoint = tmp_path / FINAL_NAME
    gate, *_ = _evidence(tmp_path)
    deployment._require_final_gate(
        gate, checkpoint=checkpoint, checkpoint_sha256="1" * 64
    )
    for rejected in (
        "full_body_reachable_performance_perturbation_v2",
        "full_body_reachable_performance_perturbation_v2_completion_allowance_v1",
        None,
    ):
        gate["tracking_profile"] = rejected
        with pytest.raises(ValueError, match="requires the passing gate"):
            deployment._require_final_gate(
                gate, checkpoint=checkpoint, checkpoint_sha256="1" * 64
            )


def test_metadata_covers_runtime_contract_and_derives_guard(tmp_path: Path) -> None:
    checkpoint = tmp_path / FINAL_NAME
    checkpoint.write_bytes(b"checkpoint")
    gate_path = tmp_path / "gate.json"
    gate_path.write_text("{}", encoding="utf-8")
    gate, locomotion, tracking, onnx_report, infos = _evidence(tmp_path)
    gate["checkpoint_sha256"] = _sha(checkpoint)
    metadata = deployment.build_v12_deployment_metadata(
        checkpoint=checkpoint,
        checkpoint_sha256=_sha(checkpoint),
        gate_path=gate_path,
        gate=gate,
        infos=infos,
        bootstrap=_bootstrap(),
        locomotion=locomotion,
        tracking=tracking,
        onnx_report=onnx_report,
        packager_parity={
            "reference_maximum_absolute_error": 1.0e-6,
            "onnxruntime_cpu_maximum_absolute_error": 2.0e-6,
        },
    )
    assert not deployment.REQUIRED_V12_RUNTIME_METADATA_KEYS.difference(metadata)
    assert json.loads(metadata["runtime_raw_action_guard_absmax_json"]) == [24.0] * 18
    assert metadata["v12_stage_gate_sha256"] == _sha(gate_path)
    assert metadata["deployment_accepted"] == "true"
    assert metadata["v12_bilateral_site_order_revision"] == (
        MICROBAN_BILATERAL_SITE_ORDER_REVISION
    )
    # The contract keys; no robot source hashes any more.
    assert metadata["policy_contract"] == POLICY_CONTRACT
    assert metadata["servo_kp"] == str(SERVO_KP_POLICY)
    assert json.loads(metadata["pico_schedule_json"]) == pico_schedule_record()
    assert metadata["checkpoint_iteration"] == str(PICO_TOTAL_UPDATES - 1)
    assert metadata["checkpoint_completed_updates"] == str(PICO_TOTAL_UPDATES)
    assert not [key for key in metadata if key.startswith("microban_") and key.endswith("_sha256")]
    assert "v12_lr_order_migration_revision" not in metadata
    assert metadata["v12_actual_dynamic_soft_limit_overshoot_max_deg"] == str(
        ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_DEG
    )
    assert metadata["v12_actual_dynamic_soft_limit_overshoot_max_rad"] == str(
        ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD
    )
    assert metadata["v12_commanded_target_soft_limit_excess_max_rad"] == str(
        COMMANDED_TARGET_SOFT_LIMIT_EXCESS_MAX_RAD
    )
    assert metadata["hand_target_lower"] == [-0.08] * 6
    assert metadata["hand_target_upper"] == [0.08] * 6
    hand_target_fk = json.loads(metadata["hand_target_fk"])
    assert hand_target_fk["joint_upper_deg"] == [
        [25.0, 30.0, -10.0],
        [25.0, -10.0, -10.0],
    ]
    from mjlab_microban.robot.microban_constants import HOME_TRUNK_PITCH_RAD
    from mjlab_microban.robot.microban_hand_fk import MICROBAN_HAND_TARGET_NORMALIZER_ABS_BOUND_M

    # The published HOMEs' boxes (hand FK v2 at the centered HOME, the levelled
    # receiver box of hand FK v4 at the forward-lean HOME); another HOME's own.
    assert hand_target_fk["normalizer_abs_bound_m"] == {
        CENTERED_HOME_TAG: [0.063, 0.0388, 0.0605],
        FORWARD_LEAN_HOME_TAG: [0.064, 0.0388, 0.0458],
    }.get(home_tag(), list(MICROBAN_HAND_TARGET_NORMALIZER_ABS_BOUND_M))
    if HOME_TRUNK_PITCH_RAD != 0.0:
        levelled = "robot_home_levelled_trunk_xyz_forward_left_up"
        assert hand_target_fk["target_frame"] == levelled
        assert metadata["foot_target_frame"] == levelled
        assert metadata["hand_target_frame"] == levelled
    assert metadata["action_clip_semantics"] == (
        "absolute_target_saturated_at_servo_goal_range_pi_no_software_clip_"
        "all_body_joints_radians"
    )
    assert metadata["action_target_semantics"] == (
        "default_joint_pos_plus_raw_action_times_scale_saturated_at_action_clip"
    )
    assert metadata["runtime_action_semantics"] == (
        "raw_default_plus_scale_then_servo_goal_range_saturation_v3"
    )
    # Written at full precision: a 3-decimal "3.142" would be wider than the
    # servo goal range the robot enforces.
    for key, sign in (("action_clip_lower", -1.0), ("action_clip_upper", 1.0)):
        assert isinstance(metadata[key], str)
        assert deployment._wire_metadata_value(metadata[key]) == metadata[key]
        assert [float(value) for value in metadata[key].split(",")] == [
            sign * math.pi
        ] * 18
    assert metadata["physical_motor_target_guard_semantics"] == (
        "finite_target_then_servo_goal_range_saturation_pi_v3"
    )


def test_deployment_requires_the_corrected_bilateral_site_order() -> None:
    infos = {BILATERAL_SITE_ORDER_INFO_KEY: MICROBAN_BILATERAL_SITE_ORDER_REVISION}
    require_bilateral_site_order(infos)
    for changed in ({}, {BILATERAL_SITE_ORDER_INFO_KEY: "raw_pre_fix"}):
        with pytest.raises(ValueError, match="bilateral site order"):
            require_bilateral_site_order(changed)


def test_hashed_report_loader_rejects_changed_evidence(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text('{"status":"pass"}', encoding="utf-8")
    expected = _sha(report)
    report.write_text('{"status":"fail"}', encoding="utf-8")
    with pytest.raises(ValueError, match="JSON SHA-256 mismatch"):
        deployment._load_json(report, expected_sha256=expected)


def test_output_cannot_replace_checkpoint_or_validator_source(tmp_path: Path) -> None:
    checkpoint = (tmp_path / FINAL_NAME).resolve()
    validator = (tmp_path / "validate_pico_policy.py").resolve()
    with pytest.raises(ValueError, match="checkpoint"):
        deployment._reject_protected_output(
            checkpoint, {"checkpoint": checkpoint, "validator": validator}
        )
    with pytest.raises(ValueError, match="validator"):
        deployment._reject_protected_output(
            validator, {"checkpoint": checkpoint, "validator": validator}
        )


def test_a_failed_final_check_preserves_last_known_good_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / FINAL_NAME
    checkpoint.write_bytes(b"captured-checkpoint")
    gate_path = tmp_path / "gate.json"
    gate_path.write_text("{}", encoding="utf-8")
    output = tmp_path / "deployed.onnx"
    output.write_bytes(b"last-known-good")
    gate, locomotion, tracking, onnx_report, infos = _evidence(tmp_path)
    gate["checkpoint_sha256"] = _sha(checkpoint)
    for name, value in (
        ("locomotion", locomotion),
        ("tracking", tracking),
        ("onnx", onnx_report),
    ):
        report_path = Path(gate["reports"][name])
        report_path.write_text(json.dumps(value), encoding="utf-8")
        gate["report_sha256"][name] = _sha(report_path)
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    class FakeActor:
        pass

    monkeypatch.setattr(deployment, "validate_gate", lambda *_args: gate)
    monkeypatch.setattr(
        deployment,
        "_load_actor",
        lambda *_args, **_kwargs: (FakeActor(), PICO_TOTAL_UPDATES - 1, infos),
    )
    monkeypatch.setattr(
        deployment,
        "validate_bootstrap_provenance",
        lambda *_args, **_kwargs: _bootstrap(),
    )
    monkeypatch.setattr(
        deployment,
        "_export_onnx_atomic",
        lambda _actor, destination: destination.write_bytes(b"candidate"),
    )
    monkeypatch.setattr(deployment, "_validate_graph_contract", lambda _path: None)
    parity_calls: list[int] = []

    def parity(*_args, **_kwargs):
        parity_calls.append(1)
        if len(parity_calls) > 1:
            raise RuntimeError("final parity rejected")
        return {
            "reference_maximum_absolute_error": 1.0e-6,
            "onnxruntime_cpu_maximum_absolute_error": 2.0e-6,
        }

    monkeypatch.setattr(deployment, "_validate_final_parity", parity)
    attached: dict[str, object] = {}

    def attach(_path: str, metadata: dict[str, object]) -> None:
        attached.update(metadata)

    monkeypatch.setattr(deployment, "attach_metadata_to_onnx", attach)
    monkeypatch.setattr(
        deployment,
        "_read_onnx_metadata",
        lambda _path: {
            key: deployment._wire_metadata_value(value)
            for key, value in attached.items()
        },
    )
    with pytest.raises(RuntimeError, match="final parity rejected"):
        deployment.package_v12_deployment(
            checkpoint=checkpoint,
            gate_path=gate_path,
            output=output,
            force=True,
        )
    assert output.read_bytes() == b"last-known-good"
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".*.captured"))


def test_runtime_smoke_corpus_comes_from_the_final_tracking_report():
    from mjlab_microban.scripts.export_teleop_v12_deployment import _runtime_smoke_corpus

    rows = [[0.0] * 5 + [-1.0] + [0.0] * 77 for _ in range(16)]
    assert _runtime_smoke_corpus({"runtime_smoke_observations": rows}) == rows
    for bad in (
        {},
        {"runtime_smoke_observations": rows[:7]},
        {"runtime_smoke_observations": rows * 5},
        {"runtime_smoke_observations": [row[:82] for row in rows]},
        {"runtime_smoke_observations": [[float("nan")] * 83] + rows[1:]},
        {"runtime_smoke_observations": [[True] * 83] + rows[1:]},
    ):
        with pytest.raises(ValueError):
            _runtime_smoke_corpus(bad)


def _normwise_onnx_evidence() -> dict[str, object]:
    return {
        "tolerance": 2.0e-5,
        "parity_rule": (
            "max_abs_error_le_atol_plus_rtol_times_max_abs_expected_per_sample_v1"
        ),
        "relative_tolerance": 1.0e-6,
        "maximum_absolute_expected_output": 46.55878829956055,
        "onnxruntime_cpu_maximum_absolute_error": 2.47955322265625e-05,
        "onnxruntime_cpu_maximum_bound_ratio": 0.451808363199234,
        "reference_evaluator_maximum_absolute_error": 9.5367431640625e-06,
        "reference_evaluator_maximum_bound_ratio": 0.1619720607995987,
    }


def test_onnx_parity_rule_metadata_ships_the_normwise_bound() -> None:
    metadata = deployment._onnx_parity_rule_metadata(_normwise_onnx_evidence())
    assert metadata == {
        "v12_onnx_parity_rule": (
            "max_abs_error_le_atol_plus_rtol_times_max_abs_expected_per_sample_v1"
        ),
        "v12_onnx_parity_relative_tolerance": "1e-06",
        "v12_onnx_parity_max_abs_expected_output": "46.55878829956055",
        "v12_onnx_reference_max_bound_ratio": "0.1619720607995987",
        "v12_onnxruntime_cpu_max_bound_ratio": "0.451808363199234",
    }
    # The robot's cap atol + rtol * magnitude covers the shipped CPU error.
    cap = 2.0e-5 + float(metadata["v12_onnx_parity_relative_tolerance"]) * float(
        metadata["v12_onnx_parity_max_abs_expected_output"]
    )
    assert 2.47955322265625e-05 <= cap


def test_onnx_parity_rule_metadata_absent_for_absolute_only_report() -> None:
    assert deployment._onnx_parity_rule_metadata({"tolerance": 2.0e-5}) == {}


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("parity_rule", "elementwise_v0"),
        ("relative_tolerance", 1.0e-5),
        ("maximum_absolute_expected_output", math.nan),
        ("onnxruntime_cpu_maximum_bound_ratio", 1.01),
        ("reference_evaluator_maximum_bound_ratio", None),
    ),
)
def test_onnx_parity_rule_metadata_rejects_drift(name: str, value: object) -> None:
    evidence = _normwise_onnx_evidence()
    evidence[name] = value
    with pytest.raises(ValueError):
        deployment._onnx_parity_rule_metadata(evidence)


def test_dry_run_evidence_is_refused_outside_dry_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(deployment, "resolve_bootstrap_artifact_path", Path)
    probe = tmp_path / "velocity_probe.json"
    probe.write_text(json.dumps({"summary": {}}), encoding="utf-8")
    assert deployment._dry_run_evidence(gate={"status": "pass"}, infos={}, probe_path=str(probe)) == []
    forced = tmp_path / "DRYRUN_FORCED_PASS_velocity_probe.json"
    forced.write_text(json.dumps({"dry_run_forced_pass_not_deployable": True}), encoding="utf-8")
    assert deployment._dry_run_evidence(gate={}, infos={}, probe_path=str(forced)) == [
        f"source probe receipt {forced.name}"
    ]
    renamed = tmp_path / "velocity_probe_copy.json"
    renamed.write_text(forced.read_text(encoding="utf-8"), encoding="utf-8")
    assert deployment._dry_run_evidence(
        gate={"dry_run_status_forced_not_deployable": True},
        infos={"dry_run_clock_lift_from": "x"},
        probe_path=str(renamed),
    ) == [
        "gate dry_run_status_forced_not_deployable",
        "checkpoint info dry_run_clock_lift_from",
        "source probe receipt dry_run_forced_pass_not_deployable",
    ]
    assert deployment.DRY_RUN_METADATA_KEY == "dry_run_not_deployable"
