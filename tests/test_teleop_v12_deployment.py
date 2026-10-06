from __future__ import annotations

import hashlib
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from home_cases import CENTERED_HOME_TAG, FORWARD_LEAN_HOME_TAG, home_tag  # noqa: E402

from mjlab_microban.scripts import export_teleop_v12_deployment as deployment
from mjlab_microban.scripts.evaluate_teleop_v12_tracking import (
    FOOT_ACTIVATION_CANARY_PROFILE,
    HMD_HAND_PROFILE,
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
    MICROBAN_TELEOP_V12_RECIPE_REVISION,
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


def _microban_identity() -> dict[str, str]:
    return {
        "microban_runtime_validator_source_sha256": "5" * 64,
        "microban_runtime_contract_source_sha256": "6" * 64,
        "microban_runtime_selector_source_sha256": "7" * 64,
        "microban_walk_runtime_source_sha256": "8" * 64,
        "microban_walk_config_source_sha256": "9" * 64,
        "microban_arm_runtime_source_sha256": "c" * 64,
        "microban_arm_contract_source_sha256": "d" * 64,
        "microban_network_input_source_sha256": "e" * 64,
        "microban_input_contract_source_sha256": "f" * 64,
        "microban_runtime_entrypoint_source_sha256": "0" * 64,
        "microban_scheduler_source_sha256": "1" * 64,
        "microban_runtime_lock_sha256": "a" * 64,
        "microban_walk_fallback_onnx_sha256": "b" * 64,
    }


def _evidence(root: Path) -> tuple[dict, dict, dict, dict, dict]:
    checkpoint_sha = "1" * 64
    reports = {
        "locomotion": str(root / "locomotion.json"),
        "tracking": str(root / "tracking.json"),
        "onnx": str(root / "onnx.json"),
    }
    gate = {
        "schema_version": 2,
        "gate": "microban_teleop_v12_stage",
        "status": "pass",
        TELEOP_V12_HOME_POSE_INFO_KEY: teleop_v12_home_pose_marker(),
        "checkpoint_sha256": checkpoint_sha,
        "iteration": 14_999,
        "completed_updates": 15_000,
        "canonical_boundary": True,
        "checkpoint_kind": "canonical_boundary",
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


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("status", "fail"),
        ("iteration", 14_998),
        ("completed_updates", 14_999),
        ("canonical_boundary", False),
        ("checkpoint_kind", "interrupted_recovery"),
        ("tracking_profile", "expanded_locomotion_tracking_v1"),
    ],
)
def test_final_gate_rejects_every_nonfinal_identity(
    tmp_path: Path, field: str, bad_value: object
) -> None:
    checkpoint = tmp_path / "model_14999.pt"
    gate, *_ = _evidence(tmp_path)
    gate[field] = bad_value
    with pytest.raises(ValueError, match="exact accepted 15000-update gate"):
        deployment._require_final_gate(
            gate,
            checkpoint=checkpoint,
            checkpoint_sha256="1" * 64,
        )


def test_final_gate_profile_is_the_one_final_profile(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_14999.pt"
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
        with pytest.raises(ValueError, match="exact accepted 15000-update gate"):
            deployment._require_final_gate(
                gate, checkpoint=checkpoint, checkpoint_sha256="1" * 64
            )


def test_metadata_covers_runtime_contract_and_derives_guard(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_14999.pt"
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
        microban_source_identity=_microban_identity(),
    )
    assert not deployment.REQUIRED_V12_RUNTIME_METADATA_KEYS.difference(metadata)
    assert json.loads(metadata["runtime_raw_action_guard_absmax_json"]) == [24.0] * 18
    assert metadata["v12_stage_gate_sha256"] == _sha(gate_path)
    assert metadata["deployment_accepted"] == "true"
    assert metadata["v12_bilateral_site_order_revision"] == (
        MICROBAN_BILATERAL_SITE_ORDER_REVISION
    )
    assert metadata["v12_lr_order_migration_revision"] == (
        deployment.LR_ORDER_NO_MIGRATION_REVISION
    )
    assert {
        name: metadata[name] for name in _microban_identity()
    } == _microban_identity()
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
    checkpoint = (tmp_path / "model_14999.pt").resolve()
    validator = (tmp_path / "validate_pico_policy.py").resolve()
    with pytest.raises(ValueError, match="checkpoint"):
        deployment._reject_protected_output(
            checkpoint, {"checkpoint": checkpoint, "validator": validator}
        )
    with pytest.raises(ValueError, match="validator"):
        deployment._reject_protected_output(
            validator, {"checkpoint": checkpoint, "validator": validator}
        )


def test_runtime_validator_requires_cpu_only_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "microban"
    runtime_files = (
        repo / "tools" / "validate_pico_policy.py",
        repo / "src" / "moves" / "pico_hybrid.py",
        repo / "src" / "moves" / "policy_selector.py",
        repo / "src" / "moves" / "walk.py",
        repo / "src" / "constants.py",
        repo / "src" / "moves" / "pico_arms.py",
        repo / "src" / "pico_arm_contract.py",
        repo / "src" / "input" / "network_input.py",
        repo / "src" / "input" / "input_source.py",
        repo / "src" / "main.py",
        repo / "src" / "scheduler.py",
        repo / "uv.lock",
        repo / "src" / "agents" / "walk.onnx",
    )
    for runtime_file in runtime_files:
        runtime_file.parent.mkdir(parents=True, exist_ok=True)
        runtime_file.write_bytes(f"fixture:{runtime_file.name}".encode())
    source_identity = deployment._microban_runtime_source_identity(repo)
    policy = tmp_path / "policy.onnx"
    policy.write_bytes(b"onnx")
    monkeypatch.setattr(deployment.shutil, "which", lambda _name: "/usr/bin/uv")

    report = {
        "status": "pass",
        "policy": str(policy.resolve()),
        "input_width": 83,
        "output_width": 18,
        "training_contract_version": "12",
        "checkpoint_iteration": 14_999,
        "checkpoint_completed_updates": 15_000,
        "checkpoint_sha256": "1" * 64,
        "v12_stage_gate_sha256": "2" * 64,
        "runtime_source_identity": source_identity,
        "onnxruntime_compatibility_smoke": {
            "status": "pass",
            "sample_count": 16,
            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        },
        "walk_fallback": {
            "status": "pass",
            "policy": str((repo / "src" / "agents" / "walk.onnx").resolve()),
            "sha256": source_identity["microban_walk_fallback_onnx_sha256"],
            "providers": ["CPUExecutionProvider"],
            "input": {"name": "obs", "shape": [1, 63], "type": "tensor(float)"},
            "output": {
                "name": "actions",
                "shape": [1, 18],
                "type": "tensor(float)",
            },
            "smoke": {
                "status": "pass",
                "sample_count": 16,
                "corpus": "deterministic_exact_float32_mod29_v1",
                "all_outputs_finite": True,
                "maximum_absolute_output": 1.0,
            },
        },
    }
    monkeypatch.setattr(
        deployment.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(report), stderr=""
        ),
    )
    monkeypatch.setattr(
        deployment,
        "_read_onnx_metadata",
        lambda _path: {
            "checkpoint_sha256": "1" * 64,
            "v12_stage_gate_sha256": "2" * 64,
            **source_identity,
        },
    )
    with pytest.raises(RuntimeError, match="complete CPU v12"):
        deployment._run_microban_runtime_validator(policy, microban_repo=repo)

    report["onnxruntime_compatibility_smoke"]["providers"] = ["CPUExecutionProvider"]
    fallback = report.pop("walk_fallback")
    with pytest.raises(RuntimeError, match="fallback pass"):
        deployment._run_microban_runtime_validator(policy, microban_repo=repo)

    report["walk_fallback"] = fallback
    accepted = deployment._run_microban_runtime_validator(policy, microban_repo=repo)
    assert accepted["runtime_source_identity"] == source_identity
    assert accepted["walk_fallback"]["status"] == "pass"


def test_runtime_rejection_preserves_last_known_good_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "model_14999.pt"
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
    microban_repo = tmp_path / "microban"
    (microban_repo / "tools").mkdir(parents=True)
    (microban_repo / "src" / "moves").mkdir(parents=True)
    (microban_repo / "tools" / "validate_pico_policy.py").write_text(
        "# validator\n", encoding="utf-8"
    )
    (microban_repo / "src" / "moves" / "pico_hybrid.py").write_text(
        "# contract\n", encoding="utf-8"
    )
    (microban_repo / "src" / "moves" / "policy_selector.py").write_text(
        "# selector\n", encoding="utf-8"
    )
    (microban_repo / "src" / "moves" / "walk.py").write_text(
        "# walk\n", encoding="utf-8"
    )
    (microban_repo / "src" / "constants.py").write_text(
        "# constants\n", encoding="utf-8"
    )
    (microban_repo / "src" / "moves" / "pico_arms.py").write_text(
        "# arm runtime\n", encoding="utf-8"
    )
    (microban_repo / "src" / "pico_arm_contract.py").write_text(
        "# arm contract\n", encoding="utf-8"
    )
    (microban_repo / "src" / "input").mkdir(parents=True)
    (microban_repo / "src" / "input" / "network_input.py").write_text(
        "# network input\n", encoding="utf-8"
    )
    (microban_repo / "src" / "input" / "input_source.py").write_text(
        "# input contract\n", encoding="utf-8"
    )
    (microban_repo / "src" / "main.py").write_text(
        "# runtime entrypoint\n", encoding="utf-8"
    )
    (microban_repo / "src" / "scheduler.py").write_text(
        "# scheduler\n", encoding="utf-8"
    )
    (microban_repo / "src" / "agents").mkdir(parents=True)
    (microban_repo / "src" / "agents" / "walk.onnx").write_bytes(b"walk")
    (microban_repo / "uv.lock").write_text("# lock\n", encoding="utf-8")

    class FakeActor:
        pass

    monkeypatch.setattr(deployment, "validate_gate", lambda *_args: gate)
    monkeypatch.setattr(
        deployment,
        "_load_actor",
        lambda *_args, **_kwargs: (FakeActor(), 14_999, infos),
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
    monkeypatch.setattr(
        deployment,
        "_validate_final_parity",
        lambda *_args, **_kwargs: {
            "reference_maximum_absolute_error": 1.0e-6,
            "onnxruntime_cpu_maximum_absolute_error": 2.0e-6,
        },
    )
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
    monkeypatch.setattr(
        deployment,
        "_run_microban_runtime_validator",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("runtime rejected")
        ),
    )
    # The fixture final has no boundary gates (the package step is what fails).
    monkeypatch.setattr(deployment, "POSE_RELEASE_REQUIRED_BOUNDARY_COMPLETED_UPDATES", ())

    with pytest.raises(RuntimeError, match="runtime rejected"):
        deployment.package_v12_deployment(
            checkpoint=checkpoint,
            gate_path=gate_path,
            output=output,
            microban_repo=microban_repo,
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


def _write_resume_params(run_dir: Path, parent: Path | None) -> None:
    """Record ``parent`` as the run's exact resume source (params/agent.yaml)."""

    params = run_dir / "params"
    params.mkdir(parents=True, exist_ok=True)
    if parent is None:
        text = "resume: false\nload_run: .*\nload_checkpoint: model_.*.pt\n"
    else:
        text = (
            "seed: 42\nresume: true\n"
            f"load_run: ^{parent.parent.name}$\n"
            f"load_checkpoint: ^{parent.stem}[.]pt$\n"
        )
    (params / "agent.yaml").write_text(text, encoding="utf-8")


def _final_checkpoint_fixture(root: Path, parent: Path) -> Path:
    checkpoint = root / "final" / "model_14999.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"final")
    _write_resume_params(checkpoint.parent, parent)
    return checkpoint


def _boundary_gate_fixture(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str,
    infos: dict,
    completed: int = 10_000,
    kind: str = "canonical_boundary",
    profile: str = HMD_HAND_PROFILE,
    parent: Path | None = None,
) -> Path:
    import torch

    checkpoint = root / name / f"model_{completed - 1}.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"iter": completed - 1, "infos": infos}, checkpoint)
    _write_resume_params(checkpoint.parent, parent)
    gate = {
        "schema_version": 2,
        "gate": "microban_teleop_v12_stage",
        "status": "pass",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha(checkpoint),
        "iteration": completed - 1,
        "completed_updates": completed,
        "canonical_boundary": kind == "canonical_boundary",
        "checkpoint_kind": kind,
        "tracking_profile": profile,
    }
    gate_path = root / name / "gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    return gate_path


def test_boundary_gates_record_the_10000_boundary_and_its_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validated: list[Path] = []

    def fake_validate_gate(gate_path: Path, checkpoint: Path) -> dict:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        assert Path(gate["checkpoint"]) == checkpoint
        validated.append(gate_path)
        return gate

    monkeypatch.setattr(deployment, "validate_gate", fake_validate_gate)
    monkeypatch.setattr(deployment, "resolve_bootstrap_artifact_path", Path)
    *_, final_infos = _evidence(tmp_path)
    final_infos["microban_teleop_recipe_revision"] = (
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
    )
    boundary_infos = deepcopy(final_infos)
    gate_path = _boundary_gate_fixture(
        tmp_path, monkeypatch, name="rescue", infos=boundary_infos
    )
    canary = _boundary_gate_fixture(
        tmp_path,
        monkeypatch,
        name="canary",
        infos=deepcopy(final_infos),
        completed=10_100,
        kind="activation_canary",
        profile=FOOT_ACTIVATION_CANARY_PROFILE,
        parent=gate_path.parent / "model_9999.pt",
    )
    final_checkpoint = _final_checkpoint_fixture(
        tmp_path, canary.parent / "model_10099.pt"
    )
    entries = deployment._boundary_stage_gate_lineage(
        (gate_path, canary), final_infos=final_infos, final_checkpoint=final_checkpoint
    )
    assert validated == [gate_path.resolve(), canary.resolve()]
    assert len(entries) == 2
    entry = entries[0]
    assert entry["completed_updates"] == 10_000
    assert entry["checkpoint_kind"] == "canonical_boundary"
    assert entry["iteration"] == 9_999
    assert entry["stage_gate_sha256"] == _sha(gate_path)
    assert entry["tracking_profile"] == HMD_HAND_PROFILE
    assert set(entry) == {
        "completed_updates",
        "checkpoint_kind",
        "iteration",
        "checkpoint_sha256",
        "stage_gate_sha256",
        "tracking_profile",
    }

    # The package carries the per-boundary profile record.
    checkpoint = tmp_path / "model_14999.pt"
    checkpoint.write_bytes(b"checkpoint")
    final_gate_path = tmp_path / "gate.json"
    final_gate_path.write_text("{}", encoding="utf-8")
    gate, locomotion, tracking, onnx_report, _ = _evidence(tmp_path)
    gate["checkpoint_sha256"] = _sha(checkpoint)
    common = {
        "checkpoint": checkpoint,
        "checkpoint_sha256": _sha(checkpoint),
        "gate_path": final_gate_path,
        "gate": gate,
        "infos": final_infos,
        "bootstrap": _bootstrap(),
        "locomotion": locomotion,
        "tracking": tracking,
        "onnx_report": onnx_report,
        "packager_parity": {
            "reference_maximum_absolute_error": 1.0e-6,
            "onnxruntime_cpu_maximum_absolute_error": 2.0e-6,
        },
        "microban_source_identity": _microban_identity(),
    }
    metadata = deployment.build_v12_deployment_metadata(
        **common, boundary_stage_gates=entries
    )
    assert metadata["v12_tracking_profile"] == deployment.FINAL_PROFILE
    assert metadata["v12_boundary_stage_gates_semantics"] == (
        deployment.BOUNDARY_STAGE_GATES_SEMANTICS
    )
    assert json.loads(metadata["v12_boundary_stage_gates_json"]) == entries
    plain = deployment.build_v12_deployment_metadata(**common)
    assert "v12_boundary_stage_gates_json" not in plain
    assert "v12_boundary_stage_gates_semantics" not in plain


def test_boundary_gates_record_the_10100_canary_and_are_discovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    monkeypatch.setattr(
        deployment,
        "validate_gate",
        lambda gate_path, _checkpoint: json.loads(gate_path.read_text("utf-8")),
    )
    monkeypatch.setattr(deployment, "resolve_bootstrap_artifact_path", Path)
    *_, final_infos = _evidence(tmp_path)
    final_infos["microban_teleop_recipe_revision"] = (
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
    )
    canary = _boundary_gate_fixture(
        tmp_path,
        monkeypatch,
        name="canary",
        infos=deepcopy(final_infos),
        completed=10_100,
        kind="activation_canary",
        profile=FOOT_ACTIVATION_CANARY_PROFILE,
    )
    boundary = _boundary_gate_fixture(
        tmp_path, monkeypatch, name="boundary", infos=deepcopy(final_infos)
    )
    _write_resume_params(canary.parent, boundary.parent / "model_9999.pt")
    final_checkpoint = _final_checkpoint_fixture(
        tmp_path, canary.parent / "model_10099.pt"
    )
    entries = deployment._boundary_stage_gate_lineage(
        (canary, boundary), final_infos=final_infos, final_checkpoint=final_checkpoint
    )
    # The required pose-release ancestors are discovered through the resume
    # chain when they are not listed explicitly.
    gate_root = tmp_path / "gates"
    gate_root.mkdir()
    for gate, stem in ((boundary, "model_9999"), (canary, "model_10099")):
        (gate_root / f"{gate.parent.name}_{stem}_gate.json").write_bytes(
            gate.read_bytes()
        )
    discovered = deployment._discover_ancestor_boundary_gates(
        final_checkpoint, gate_root=gate_root, explicit=()
    )
    assert sorted(path.name for path in discovered) == [
        "boundary_model_9999_gate.json",
        "canary_model_10099_gate.json",
    ]
    assert (
        deployment._boundary_stage_gate_lineage(
            discovered, final_infos=final_infos, final_checkpoint=final_checkpoint
        )
        == entries
    )
    assert deployment._discover_ancestor_boundary_gates(
        final_checkpoint, gate_root=gate_root, explicit=(boundary,)
    ) == (boundary, (gate_root / "canary_model_10099_gate.json").resolve())
    assert [entry["completed_updates"] for entry in entries] == [10_000, 10_100]
    assert [entry["checkpoint_kind"] for entry in entries] == [
        "canonical_boundary",
        "activation_canary",
    ]
    assert entries[1]["tracking_profile"] == FOOT_ACTIVATION_CANARY_PROFILE


@pytest.mark.parametrize(
    "case",
    [
        "other_recipe",
        "final_clock",
        "interrupted_recovery",
        "duplicate_clock",
        "sibling_not_ancestor",
        "missing_canary",
        "broken_resume_record",
    ],
)
def test_boundary_gates_reject_foreign_or_nonboundary_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    monkeypatch.setattr(
        deployment,
        "validate_gate",
        lambda gate_path, _checkpoint: json.loads(gate_path.read_text("utf-8")),
    )
    monkeypatch.setattr(deployment, "resolve_bootstrap_artifact_path", Path)
    *_, final_infos = _evidence(tmp_path)
    final_infos["microban_teleop_recipe_revision"] = (
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
    )
    boundary_infos = deepcopy(final_infos)
    fixture: dict = {}
    if case == "other_recipe":
        boundary_infos["microban_teleop_recipe_revision"] = (
            MICROBAN_TELEOP_V12_RECIPE_REVISION
        )
    elif case == "final_clock":
        fixture = {"completed": 15_000}
    elif case == "interrupted_recovery":
        fixture = {
            "completed": 10_050,
            "kind": "interrupted_recovery",
            "profile": "whole_body_foot_activation_canary_reachable_safety_v1",
        }

    gates = [
        _boundary_gate_fixture(
            tmp_path, monkeypatch, name="a", infos=boundary_infos, **fixture
        )
    ]
    canary = _boundary_gate_fixture(
        tmp_path,
        monkeypatch,
        name="canary",
        infos=deepcopy(final_infos),
        completed=10_100,
        kind="activation_canary",
        profile=FOOT_ACTIVATION_CANARY_PROFILE,
        parent=gates[0].parent / f"model_{fixture.get('completed', 10_000) - 1}.pt",
    )
    final_checkpoint = _final_checkpoint_fixture(
        tmp_path, canary.parent / "model_10099.pt"
    )
    if case != "missing_canary":
        gates.append(canary)
    if case == "duplicate_clock":
        gates.append(
            _boundary_gate_fixture(tmp_path, monkeypatch, name="b", infos=boundary_infos)
        )
        _write_resume_params(gates[0].parent, gates[-1].parent / "model_9999.pt")
    if case == "sibling_not_ancestor":
        # Same markers and clock, but not on the final's resume chain.
        gates[0] = _boundary_gate_fixture(
            tmp_path, monkeypatch, name="sibling", infos=boundary_infos
        )
    if case == "broken_resume_record":
        (canary.parent / "params" / "agent.yaml").write_text(
            "resume: true\nload_run: .*\nload_checkpoint: model_.*.pt\n",
            encoding="utf-8",
        )
    with pytest.raises(ValueError):
        deployment._boundary_stage_gate_lineage(
            tuple(gates), final_infos=final_infos, final_checkpoint=final_checkpoint
        )


def test_boundary_gate_controls_pass_on_the_exact_ancestry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rejection fixture above is valid except for each injected defect."""


    monkeypatch.setattr(
        deployment,
        "validate_gate",
        lambda gate_path, _checkpoint: json.loads(gate_path.read_text("utf-8")),
    )
    monkeypatch.setattr(deployment, "resolve_bootstrap_artifact_path", Path)
    *_, final_infos = _evidence(tmp_path)
    final_infos["microban_teleop_recipe_revision"] = (
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
    )
    boundary = _boundary_gate_fixture(
        tmp_path, monkeypatch, name="a", infos=deepcopy(final_infos)
    )
    canary = _boundary_gate_fixture(
        tmp_path,
        monkeypatch,
        name="canary",
        infos=deepcopy(final_infos),
        completed=10_100,
        kind="activation_canary",
        profile=FOOT_ACTIVATION_CANARY_PROFILE,
        parent=boundary.parent / "model_9999.pt",
    )
    final_checkpoint = _final_checkpoint_fixture(
        tmp_path, canary.parent / "model_10099.pt"
    )
    entries = deployment._boundary_stage_gate_lineage(
        (boundary, canary), final_infos=final_infos, final_checkpoint=final_checkpoint
    )
    assert [entry["completed_updates"] for entry in entries] == [10_000, 10_100]
    # A non-pose-release final may still be packaged without boundary gates.
    final_infos["microban_teleop_recipe_revision"] = (
        MICROBAN_TELEOP_V12_RECIPE_REVISION
    )
    assert (
        deployment._boundary_stage_gate_lineage(
            (), final_infos=final_infos, final_checkpoint=final_checkpoint
        )
        == []
    )


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
