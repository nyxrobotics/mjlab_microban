from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from home_cases import FORWARD_LEAN_HOME_TAG, home_tag  # noqa: E402

from mjlab_microban.scripts import export_teleop_v12_deployment as deployment
from mjlab_microban.policy_contract import POLICY_CONTRACT
from mjlab_microban.schedules import (
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
    TELEOP_TRAINABLE_ACTOR_PARAMETERS,
    TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
)
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import (
    LegacyTeleopProbeIdentity,
    LegacyVelocitySourceIdentity,
    TeleopV12BootstrapProvenance,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_ACTION_CLIP,
    MICROBAN_TELEOP_V13_ARM_OVERLAY_RECIPE_REVISION,
)
from mjlab_microban.tasks.microban_teleop_v12_home_pose import (
    TELEOP_V12_HOME_POSE_INFO_KEY,
    teleop_v12_home_pose_marker,
)
from mjlab_microban.tasks.microban_teleop_v12_runner import (
    BILATERAL_SITE_ORDER_INFO_KEY,
    require_bilateral_site_order,
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
            path="/runs/velocity/model_7999.pt",
            sha256=LEGACY_VELOCITY_CHECKPOINT_SHA256,
            iteration=7_999,
            normalizer_count=786_432_000,
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
        "runtime_smoke_observations": [_smoke_row(row) for row in range(16)],
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
        "microban_teleop_recipe_revision": MICROBAN_TELEOP_V13_ARM_OVERLAY_RECIPE_REVISION,
        TELEOP_V12_HOME_POSE_INFO_KEY: teleop_v12_home_pose_marker(),
        "trainable_actor_parameters": list(TELEOP_TRAINABLE_ACTOR_PARAMETERS),
        "trainable_actor_columns": list(TELEOP_V12_EXTRA_OBSERVATION_COLUMNS),
        "residual_hidden_dims": [64, 64],
        "active_actor_columns_at_save": list(TELEOP_V12_EXTRA_OBSERVATION_COLUMNS),
        deployment.TELEOP_V12_BOOTSTRAP_INFO_KEY: {},
        BILATERAL_SITE_ORDER_INFO_KEY: MICROBAN_BILATERAL_SITE_ORDER_REVISION,
    }
    return gate, locomotion, tracking, onnx_report, infos


FINAL_NAME = f"model_{PICO_TOTAL_UPDATES - 1}.pt"


def _smoke_row(row: int) -> list[float]:
    """A possible PICO observation: a lifted foot and moved arms (in the box)."""

    value = 0.001 * row
    arms = [value, value, value, value, -value, value]
    return [0.0] * 5 + [-1.0] + [value] * 63 + [value] * 6 + arms


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("status", "fail"),
        ("schema_version", 2),
        ("iteration", PICO_TOTAL_UPDATES - 2),
        ("completed_updates", PICO_TOTAL_UPDATES - 1),
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


def test_the_packaged_checkpoint_is_the_gates(tmp_path: Path) -> None:
    gate, *_ = _evidence(tmp_path)
    deployment._require_final_gate(gate, checkpoint=tmp_path / FINAL_NAME, checkpoint_sha256="1" * 64)
    with pytest.raises(ValueError, match="model_<iteration>"):
        deployment._require_final_gate(
            gate, checkpoint=tmp_path / f"model_{PICO_TOTAL_UPDATES - 1000}.pt", checkpoint_sha256="1" * 64
        )


def test_final_gate_profile_is_the_one_final_profile(tmp_path: Path) -> None:
    checkpoint = tmp_path / FINAL_NAME
    gate, *_ = _evidence(tmp_path)
    deployment._require_final_gate(
        gate, checkpoint=checkpoint, checkpoint_sha256="1" * 64
    )
    for rejected in (
        "another_profile",
        None,
    ):
        gate["tracking_profile"] = rejected
        with pytest.raises(ValueError, match="requires the passing gate"):
            deployment._require_final_gate(
                gate, checkpoint=checkpoint, checkpoint_sha256="1" * 64
            )


def test_metadata_is_the_pico_contract_and_derives_guard(tmp_path: Path) -> None:
    checkpoint = tmp_path / FINAL_NAME
    checkpoint.write_bytes(b"checkpoint")
    gate_path = tmp_path / "gate.json"
    gate_path.write_text("{}", encoding="utf-8")
    gate, _locomotion, tracking, _onnx_report, infos = _evidence(tmp_path)
    gate["checkpoint_sha256"] = _sha(checkpoint)
    rows = deployment._self_test_rows(tracking)
    metadata = deployment.build_v12_deployment_metadata(
        checkpoint=checkpoint,
        checkpoint_sha256=_sha(checkpoint),
        gate_path=gate_path,
        gate=gate,
        infos=infos,
        bootstrap=_bootstrap(),
        tracking=tracking,
        self_test_observations=rows,
        self_test_actions=[[0.5] * 18 for _ in rows],
        dry_run=False,
    )
    from test_policy_contract import ROBOT_COMMON_KEYS, ROBOT_PICO_KEYS

    assert set(metadata) == ROBOT_COMMON_KEYS | ROBOT_PICO_KEYS | {"run_path"}
    assert all(isinstance(value, str) for value in metadata.values())
    assert metadata["microban_policy_contract"] == POLICY_CONTRACT
    assert metadata["microban_policy_kind"] == "pico"
    assert metadata["gate_report_sha256"] == _sha(gate_path)
    assert metadata["checkpoint_iteration"] == str(PICO_TOTAL_UPDATES - 1)
    assert metadata["previous_action_semantics"] == "raw_policy_output"
    assert json.loads(metadata["pico_raw_action_guard_json"]) == [24.0] * 18
    assert json.loads(metadata["pico_curriculum_json"]) == pico_schedule_record()
    assert metadata["pico_walk_checkpoint_sha256"] == LEGACY_VELOCITY_CHECKPOINT_SHA256
    assert "pico_hand_target_lower_json" not in metadata
    assert json.loads(metadata["pico_arm_target_json"])["slew_rad_s"] == 4.0
    assert json.loads(metadata["self_test_observations_json"]) == rows
    assert metadata["joint_names"].split(",")[:3] == ["head", "neck_roll", "neck_pitch"]
    defaults = [float(value) for value in metadata["default_joint_pos"].split(",")]
    assert len(defaults) == 21
    if home_tag() == FORWARD_LEAN_HOME_TAG:
        assert metadata["pico_target_frame"] == "robot_home_levelled_trunk_xyz_forward_left_up"
    for key in ("action_clip_lower", "action_clip_upper"):
        assert [abs(float(value)) for value in metadata[key].split(",")] == [math.pi] * 18


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
    monkeypatch.setattr(deployment, "_actor_outputs", lambda _actor, rows: [[0.0] * 18 for _ in rows])
    attached: dict[str, object] = {}

    def attach(_path: str, metadata: dict[str, object]) -> None:
        attached.update(metadata)

    monkeypatch.setattr(deployment, "attach_metadata_to_onnx", attach)
    monkeypatch.setattr(
        deployment,
        "_read_onnx_metadata",
        lambda _path: dict(attached),
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


def test_self_test_rows_are_the_final_tracking_observations():
    rows = [[0.0] * 5 + [-1.0] + [0.0] * 75 for _ in range(16)]
    assert deployment._self_test_rows({"runtime_smoke_observations": rows}) == rows
    fast = [0.0] * 5 + [-1.0] + [0.0] * 21 + [20.0] * 21 + [0.0] * 33
    assert deployment._self_test_rows({"runtime_smoke_observations": rows[:8] + [fast]}) == rows[:8]
    for bad in (
        {},
        {"runtime_smoke_observations": rows[:7]},
        {"runtime_smoke_observations": [row[:80] for row in rows]},
        {"runtime_smoke_observations": [[float("nan")] * 81] * 16},
        {"runtime_smoke_observations": [[True] * 81] + rows[1:]},
    ):
        with pytest.raises(ValueError):
            deployment._self_test_rows(bad)


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
