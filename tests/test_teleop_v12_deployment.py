from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from mjlab_microban.scripts import export_teleop_v12_deployment as deployment
from mjlab_microban.scripts.teleop_v12_lr_recovery import (
    PINNED_RAW_MODEL_9200_SHA256,
    PINNED_SOURCE_COMMON_STEP_COUNTER,
    PINNED_SOURCE_COMPLETED_UPDATES,
    PINNED_SOURCE_ITERATION,
)
from mjlab_microban.tasks.mdp import MICROBAN_BILATERAL_SITE_ORDER_REVISION
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
)
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    LEGACY_TO_TELEOP_OBSERVATION_INDEX,
    LEGACY_VELOCITY_ACTOR_STATE_KEYS,
    LEGACY_VELOCITY_CHECKPOINT_SHA256,
    LEGACY_VELOCITY_NORMALIZER_EPS,
    TELEOP_V12_ACTOR_TOPOLOGY,
    TELEOP_V12_BOOTSTRAP_MAPPING_VERSION,
    TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
)
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import (
    PINNED_LEGACY_TELEOP_PROBE_SHA256,
    LegacyTeleopProbeIdentity,
    LegacyVelocitySourceIdentity,
    TeleopV12BootstrapProvenance,
)
from mjlab_microban.tasks.microban_teleop_v12_deadline_fallback import (
    deadline_fallback_marker,
    deadline_post_canary_marker,
)
from mjlab_microban.tasks.microban_teleop_v12_lr_order import (
    ACTOR_PERMUTATION,
    ACTOR_SWAP_BLOCKS,
    BILATERAL_SITE_ORDER_INFO_KEY,
    CRITIC_PERMUTATION,
    CRITIC_SWAP_BLOCKS,
    MIGRATION_INFO_KEY,
    MIGRATION_REVISION,
)
from mjlab_microban.teleop_v12_safety import (
    ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_DEG,
    ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD,
    COMMANDED_TARGET_SOFT_LIMIT_EXCESS_MAX_RAD,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap() -> TeleopV12BootstrapProvenance:
    return TeleopV12BootstrapProvenance(
        schema_version=1,
        source=LegacyVelocitySourceIdentity(
            path="repo://checkpoints/xc330_velocity/model_14999.pt",
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
        action_clip=None,
        frozen_tensors=tuple(
            sorted(LEGACY_VELOCITY_ACTOR_STATE_KEYS - {"mlp.0.weight"})
        ),
    )


def _lr_order_marker(*, strategy: str = "swap") -> dict:
    partial_names = {
        "actor_state_dict.mlp.0.weight",
        "actor_state_dict.obs_normalizer._mean",
        "actor_state_dict.obs_normalizer._var",
        "actor_state_dict.obs_normalizer._std",
        "critic_state_dict.mlp.0.weight",
        "critic_state_dict.obs_normalizer._mean",
        "critic_state_dict.obs_normalizer._var",
        "critic_state_dict.obs_normalizer._std",
        "optimizer_state_dict.state.1.exp_avg",
        "optimizer_state_dict.state.1.exp_avg_sq",
        "optimizer_state_dict.state.9.exp_avg",
        "optimizer_state_dict.state.9.exp_avg_sq",
    }
    partial = {
        name: {
            "untouched_source_sha256": "1" * 64,
            "untouched_output_sha256": "1" * 64,
            "source_full_sha256": "2" * 64,
            "output_full_sha256": "3" * 64,
        }
        for name in partial_names
    }
    return {
        "schema_version": 1,
        "revision": MIGRATION_REVISION,
        "site_order_revision": MICROBAN_BILATERAL_SITE_ORDER_REVISION,
        "strategy": strategy,
        "source_checkpoint_path": "/pinned/model_9200.pt",
        "source_checkpoint_sha256": PINNED_RAW_MODEL_9200_SHA256,
        "source_clock": {
            "iteration": PINNED_SOURCE_ITERATION,
            "completed_updates": PINNED_SOURCE_COMPLETED_UPDATES,
            "common_step_counter": PINNED_SOURCE_COMMON_STEP_COUNTER,
        },
        "actor_w0_optimizer_parameter_id": 1,
        "critic_w0_optimizer_parameter_id": 9,
        "actor_swap_blocks": [list(block) for block in ACTOR_SWAP_BLOCKS],
        "critic_swap_blocks": [list(block) for block in CRITIC_SWAP_BLOCKS],
        "actor_permutation": list(ACTOR_PERMUTATION),
        "critic_permutation": list(CRITIC_PERMUTATION),
        "zeroed_actor_columns": [] if strategy == "swap" else list(range(75, 83)),
        "foot_adapter_at_source": {
            "active": False,
            "maximum_absolute_w0": 0.0,
            "maximum_absolute_adam_moment": 0.0,
            "handling": "unlearned_exact_zero_left_untouched",
        },
        "tensor_integrity": {
            "passed": True,
            "unchanged_tensor_count": 0,
            "unchanged_tensors": {},
            "partially_transformed_tensors": partial,
        },
    }


def _microban_identity() -> dict[str, str]:
    return {
        "microban_runtime_validator_source_sha256": "5" * 64,
        "microban_runtime_contract_source_sha256": "6" * 64,
        "microban_runtime_selector_source_sha256": "7" * 64,
        "microban_walk_runtime_source_sha256": "8" * 64,
        "microban_walk_config_source_sha256": "9" * 64,
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
        }
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
        "trainable_actor_parameters": ["mlp.0.weight"],
        "trainable_actor_columns": list(TELEOP_V12_EXTRA_OBSERVATION_COLUMNS),
        "active_actor_columns_at_save": list(TELEOP_V12_EXTRA_OBSERVATION_COLUMNS),
        deployment.TELEOP_V12_BOOTSTRAP_INFO_KEY: {},
        BILATERAL_SITE_ORDER_INFO_KEY: MICROBAN_BILATERAL_SITE_ORDER_REVISION,
        MIGRATION_INFO_KEY: _lr_order_marker(),
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


def test_deadline_final_gate_profile_is_exactly_lineage_bound(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_14999.pt"
    gate, *_ = _evidence(tmp_path)
    canonical_infos: dict[str, object] = {}
    assert (
        deployment._expected_final_tracking_profile(canonical_infos)
        == deployment.FINAL_PROFILE
    )

    deadline_infos = {
        deployment.MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_INFO_KEY: (
            deadline_fallback_marker()
        ),
        deployment.MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_INFO_KEY: (
            deadline_post_canary_marker()
        ),
    }
    expected = deployment._expected_final_tracking_profile(deadline_infos)
    assert expected == deployment.DEADLINE_FINAL_FALLBACK_PROFILE
    gate["tracking_profile"] = expected
    deployment._require_final_gate(
        gate,
        checkpoint=checkpoint,
        checkpoint_sha256="1" * 64,
        expected_tracking_profile=expected,
    )

    gate["tracking_profile"] = deployment.FINAL_PROFILE
    with pytest.raises(ValueError, match="exact accepted 15000-update gate"):
        deployment._require_final_gate(
            gate,
            checkpoint=checkpoint,
            checkpoint_sha256="1" * 64,
            expected_tracking_profile=expected,
        )

    missing_post = {
        deployment.MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_INFO_KEY: (
            deadline_fallback_marker()
        )
    }
    with pytest.raises(ValueError, match="missing post-canary lineage"):
        deployment._expected_final_tracking_profile(missing_post)


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
    assert json.loads(metadata["runtime_raw_action_guard_absmax_json"]) == [8.0] * 18
    assert metadata["v12_stage_gate_sha256"] == _sha(gate_path)
    assert metadata["deployment_accepted"] == "true"
    assert metadata["v12_bilateral_site_order_revision"] == (
        MICROBAN_BILATERAL_SITE_ORDER_REVISION
    )
    assert metadata["v12_lr_order_migration_strategy"] == "swap"
    assert metadata["v12_lr_order_source_checkpoint_sha256"] == (
        PINNED_RAW_MODEL_9200_SHA256
    )
    assert metadata["v12_lr_order_source_checkpoint_iteration"] == "9200"
    assert metadata["v12_lr_order_source_completed_updates"] == "9201"
    assert metadata["v12_lr_order_source_common_step_counter"] == "220824"
    assert len(metadata["v12_lr_order_migration_marker_sha256"]) == 64
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
    assert hand_target_fk["normalizer_abs_bound_m"] == [0.063, 0.0388, 0.0605]


def test_deployment_requires_exact_corrected_bilateral_lineage() -> None:
    infos = {
        BILATERAL_SITE_ORDER_INFO_KEY: MICROBAN_BILATERAL_SITE_ORDER_REVISION,
        MIGRATION_INFO_KEY: _lr_order_marker(),
    }
    marker = deployment._require_deployable_lr_order_lineage(infos)
    assert marker["source_checkpoint_sha256"] == PINNED_RAW_MODEL_9200_SHA256

    for case, mutate in (
        ("raw_pre_fix", lambda value: value.clear()),
        ("fresh_without_migration", lambda value: value.pop(MIGRATION_INFO_KEY)),
        (
            "missing_top_level_revision",
            lambda value: value.pop(BILATERAL_SITE_ORDER_INFO_KEY),
        ),
        (
            "diagnostic_zero_hand",
            lambda value: value.__setitem__(
                MIGRATION_INFO_KEY, _lr_order_marker(strategy="zero_hand")
            ),
        ),
        (
            "wrong_pinned_source",
            lambda value: value[MIGRATION_INFO_KEY].__setitem__(
                "source_checkpoint_sha256", "0" * 64
            ),
        ),
    ):
        changed = deepcopy(infos)
        mutate(changed)
        with pytest.raises(
            (TypeError, ValueError), match="bilateral|predates|migration"
        ):
            deployment._require_deployable_lr_order_lineage(changed)


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
        lambda *_args: {
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
