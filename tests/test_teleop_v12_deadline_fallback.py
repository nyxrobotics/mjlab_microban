"""CPU-only fail-closed tests for the recorded deadline fallback route."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest
import torch

from mjlab_microban.scripts.evaluate_teleop_v12_tracking import (
    DEADLINE_CANARY_FALLBACK_PROFILE,
    DEADLINE_FALLBACK_PROFILE,
    HMD_HAND_PROFILE,
    TARGET_COLUMN_ABLATION_ACTION_DELTA_MIN,
    _acceptance,
    target_column_ablation_observation_columns,
)
from mjlab_microban.scripts.teleop_v12_deadline_fallback import (
    DEADLINE_FALLBACK_RECEIPT_SCHEMA_VERSION,
    DEADLINE_POST_CANARY_RECEIPT_GATE,
    validate_deadline_post_canary_receipt_payload,
)
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION,
    TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
    TELEOP_V12_FOOT_OBSERVATION_COLUMNS,
    TELEOP_V12_TARGET_POSITION_NORMALIZER_STORED_STD,
)
from mjlab_microban.tasks.microban_teleop_v12_corner_rescue import (
    MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY,
    MICROBAN_TELEOP_V12_CORNER_RESCUE_RECIPE_REVISION,
    canonical_json_sha256,
    corner_rescue_marker,
)
from mjlab_microban.tasks.microban_teleop_v12_deadline_fallback import (
    MICROBAN_TELEOP_V12_DEADLINE_CANARY_COMMON_STEP,
    MICROBAN_TELEOP_V12_DEADLINE_CANARY_ITERATION,
    MICROBAN_TELEOP_V12_DEADLINE_CANARY_OPTIMIZER_STEP,
    MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_INFO_KEY,
    MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_RECEIPT_SHA256,
    MICROBAN_TELEOP_V12_DEADLINE_RESUME_SOURCE_INFO_KEY,
    deadline_fallback_marker,
    deadline_fallback_resume_source,
    deadline_post_canary_marker,
    validate_deadline_fallback_canary_payload,
    validate_deadline_fallback_checkpoint_payload,
    validate_deadline_fallback_marker,
    validate_deadline_fallback_resume_payload,
    validate_deadline_fallback_save_endpoint,
    validate_deadline_fallback_training_request,
    validate_deadline_post_canary_marker,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_RECIPE_REVISION,
)
from mjlab_microban.teleop_v12_safety import (
    ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD,
)

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "scripts/evaluate_microban_teleop_v12_deadline_fallback.sh"
CANARY_EVALUATOR = ROOT / "scripts/evaluate_microban_teleop_v12_deadline_canary.sh"
TRAINER = ROOT / "scripts/train_microban_teleop_v12.sh"
SELECTED_SHA = "1" * 64
CANARY_SHA = "2" * 64
STRICT_SHA = "3" * 64
CANARY_STRICT_SHA = "4" * 64
METRICS = {"maximum_rms": 0.0335, "maximum_p95": 0.041}


def _corner_marker() -> dict:
    return corner_rescue_marker(
        parent_checkpoint_sha256="5" * 64,
        parent_strict_tracking_report_sha256="6" * 64,
    )


def _marker() -> dict:
    return deadline_fallback_marker(
        selected_checkpoint_sha256=SELECTED_SHA,
        corner_marker_sha256=canonical_json_sha256(_corner_marker()),
        strict_report_sha256=STRICT_SHA,
        strict_hand_tracking_m=METRICS,
    )


def _post_canary_marker(failed: list[str] | None = None) -> dict:
    return deadline_post_canary_marker(
        canary_checkpoint_sha256=CANARY_SHA,
        strict_report_sha256=CANARY_STRICT_SHA,
        strict_failed_checks=["hand_tracking_rms"] if failed is None else failed,
        strict_hand_tracking_m=METRICS,
    )


def _optimizer(step: int) -> dict:
    first = torch.ones(512, 83)
    second = torch.ones(512, 83)
    first[:, TELEOP_V12_FOOT_OBSERVATION_COLUMNS] = 0.0
    second[:, TELEOP_V12_FOOT_OBSERVATION_COLUMNS] = 0.0
    return {
        "state": {
            1: {
                "step": torch.tensor(float(step)),
                "exp_avg": first,
                "exp_avg_sq": second,
            },
            2: {
                "step": torch.tensor(float(step)),
                "exp_avg": torch.ones(18),
                "exp_avg_sq": torch.ones(18),
            },
        }
    }


def _source_payload() -> dict:
    stored_std = torch.tensor(TELEOP_V12_TARGET_POSITION_NORMALIZER_STORED_STD)
    mean = torch.zeros(1, 83)
    var = torch.ones(1, 83)
    std = torch.ones(1, 83)
    var[:, 69:75] = stored_std[:6].square()
    std[:, 69:75] = stored_std[:6]
    weight = torch.ones(512, 83)
    weight[:, TELEOP_V12_FOOT_OBSERVATION_COLUMNS] = 0.0
    return {
        "iter": 9_999,
        "actor_state_dict": {
            "obs_normalizer._mean": mean,
            "obs_normalizer._var": var,
            "obs_normalizer._std": std,
            "mlp.0.weight": weight,
        },
        "optimizer_state_dict": _optimizer(200_000),
        "infos": {
            "microban_teleop_training_contract_version": "12",
            "microban_teleop_recipe_revision": (
                MICROBAN_TELEOP_V12_CORNER_RESCUE_RECIPE_REVISION
            ),
            "adapter_gradient_schedule_revision": (
                TELEOP_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION
            ),
            "active_actor_columns_at_save": [
                6,
                7,
                8,
                27,
                28,
                29,
                75,
                76,
                77,
                78,
                79,
                80,
                81,
                82,
            ],
            "env_state": {"common_step_counter": 240_000},
            MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY: _corner_marker(),
        },
    }


def _canary_payload(*, checkpoint_path: str, gate_path: str, gate_sha: str) -> dict:
    return {
        "iter": MICROBAN_TELEOP_V12_DEADLINE_CANARY_ITERATION,
        "optimizer_state_dict": _optimizer(
            MICROBAN_TELEOP_V12_DEADLINE_CANARY_OPTIMIZER_STEP
        ),
        "infos": {
            "microban_teleop_recipe_revision": MICROBAN_TELEOP_V12_RECIPE_REVISION,
            "active_actor_columns_at_save": list(TELEOP_V12_EXTRA_OBSERVATION_COLUMNS),
            "env_state": {
                "common_step_counter": MICROBAN_TELEOP_V12_DEADLINE_CANARY_COMMON_STEP
            },
            MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY: _corner_marker(),
            MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_INFO_KEY: _marker(),
            MICROBAN_TELEOP_V12_DEADLINE_RESUME_SOURCE_INFO_KEY: (
                deadline_fallback_resume_source(
                    checkpoint_path=checkpoint_path,
                    gate_path=gate_path,
                    gate_sha256=gate_sha,
                    parent_checkpoint_sha256=SELECTED_SHA,
                )
            ),
        },
    }


def _ablation(target: str, expected: bool) -> dict:
    ablated, preserved = target_column_ablation_observation_columns(target)
    return {
        "target_expected": expected,
        "ablated_observation_columns": list(ablated),
        "preserved_observation_columns": list(preserved),
        "maximum_absolute_action_delta": 0.01 if expected else None,
        "minimum_required_action_delta": (
            TARGET_COLUMN_ABLATION_ACTION_DELTA_MIN if expected else None
        ),
        "passed": True,
    }


def _tracking_result(*, rms: float, p95: float, soft_limit: float = 0.0) -> dict:
    return {
        "completed": True,
        "fell": False,
        "nonfinite": None,
        "maximum_actual_soft_limit_violation_rad": soft_limit,
        "raw_action_recurrence_verified_steps": 300,
        "executed_steps": 300,
        "hmd_motion_evidence_passed": True,
        "observation_coverage": {"passed": True},
        "twist_directional_response_passed": True,
        "target_error": {
            "active_hand": {"sample_count": 500, "rms": rms, "p95": p95},
            "foot": {"sample_count": 0, "rms": None, "p95": None},
        },
        "command": {
            "foot_target": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "hand_active": [True, True],
        },
        "target_column_ablation": {
            "hand": _ablation("hand", True),
            "foot": _ablation("foot", False),
        },
    }


def test_selected_rescue_is_structurally_eligible() -> None:
    payload = _source_payload()
    assert (
        validate_deadline_fallback_checkpoint_payload(
            payload, checkpoint_sha256=SELECTED_SHA
        )
        == _corner_marker()
    )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        validate_deadline_fallback_checkpoint_payload(
            payload, checkpoint_sha256="F" * 64
        )
    drifted = copy.deepcopy(payload)
    drifted["infos"]["microban_teleop_recipe_revision"] = (
        MICROBAN_TELEOP_V12_RECIPE_REVISION
    )
    with pytest.raises(ValueError, match="contract/clock drifted"):
        validate_deadline_fallback_checkpoint_payload(
            drifted, checkpoint_sha256=SELECTED_SHA
        )
    drifted = copy.deepcopy(payload)
    drifted["infos"][MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY]["revision"] = "x"
    with pytest.raises(ValueError, match="lineage marker drifted"):
        validate_deadline_fallback_checkpoint_payload(
            drifted, checkpoint_sha256=SELECTED_SHA
        )


def test_markers_rebuild_from_recorded_fields_and_are_tamper_evident() -> None:
    marker = _marker()
    assert validate_deadline_fallback_marker(marker) == marker
    assert marker["threshold_change"]["hand_rms_m_max"] == 0.035
    assert marker["threshold_change"]["hand_p95_m_max"] == 0.05
    assert "pending_user_review" in marker["authorization"]
    for path, value in (
        (("threshold_change", "hand_rms_m_max"), 0.04),
        (("strict_failed_checks",), ["hand_tracking_rms", "no_falls"]),
        (("selected_checkpoint", "sha256"), "nothex"),
        (("strict_hand_tracking_m", "maximum_rms"), -1.0),
    ):
        tampered = copy.deepcopy(marker)
        target = tampered
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValueError, match="lineage marker drifted"):
            validate_deadline_fallback_marker(tampered)
    post = _post_canary_marker()
    assert validate_deadline_post_canary_marker(post) == post
    assert validate_deadline_post_canary_marker(_post_canary_marker([]))[
        "canonical_failed_checks"
    ] == []
    with pytest.raises(ValueError, match="only hand_tracking_rms"):
        _post_canary_marker(["no_falls"])
    tampered = copy.deepcopy(post)
    tampered["next_endpoint"]["completed_updates"] = 16_000
    with pytest.raises(ValueError, match="post-canary authorization marker drifted"):
        validate_deadline_post_canary_marker(tampered)


def test_only_rms_is_relaxed_p95_and_safety_are_unchanged() -> None:
    result = _tracking_result(rms=0.034, p95=0.049)
    fallback_checks, fallback_status = _acceptance([result], DEADLINE_FALLBACK_PROFILE)
    strict_checks, strict_status = _acceptance([result], HMD_HAND_PROFILE)
    assert fallback_status == "pass"
    assert fallback_checks["hand_tracking_rms"] is True
    assert strict_status == "fail"
    assert strict_checks["hand_tracking_rms"] is False

    p95_checks, p95_status = _acceptance(
        [_tracking_result(rms=0.034, p95=0.050001)],
        DEADLINE_FALLBACK_PROFILE,
    )
    assert p95_status == "fail"
    assert p95_checks["hand_tracking_p95"] is False
    safety_checks, safety_status = _acceptance(
        [
            _tracking_result(
                rms=0.034,
                p95=0.049,
                soft_limit=ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD + 1.0e-9,
            )
        ],
        DEADLINE_FALLBACK_PROFILE,
    )
    assert safety_status == "fail"
    assert safety_checks["actual_soft_limits"] is False


def test_canary_fallback_keeps_foot_causality_and_relaxes_only_hand_rms() -> None:
    result = _tracking_result(rms=0.034, p95=0.049)
    result["command"]["foot_target"] = [[0.01, 0.0, 0.0], [0.0, 0.0, 0.0]]
    result["target_error"]["foot"] = {
        "sample_count": 250,
        "rms": 0.2,
        "p95": 0.3,
    }
    result["target_column_ablation"]["foot"] = _ablation("foot", True)
    checks, status = _acceptance([result], DEADLINE_CANARY_FALLBACK_PROFILE)
    assert status == "pass"
    assert checks["hand_tracking_rms"] is True
    assert checks["hand_tracking_p95"] is True
    assert checks["target_column_ablation_response"] is True
    assert "foot_tracking_rms" not in checks
    assert "foot_tracking_p95" not in checks

    result["target_column_ablation"]["foot"]["passed"] = False
    checks, status = _acceptance([result], DEADLINE_CANARY_FALLBACK_PROFILE)
    assert status == "fail"
    assert checks["target_column_ablation_response"] is False


def test_post_canary_receipt_payload_is_exact_and_fail_closed() -> None:
    checks = {
        name: True
        for name in (
            "actual_soft_limits",
            "all_scenarios_completed",
            "finite",
            "forced_hmd_motion",
            "hand_tracking_p95",
            "hand_tracking_rms",
            "no_falls",
            "nonzero_observation_coverage",
            "raw_action_recurrence",
            "target_column_ablation_response",
            "twist_directional_response",
        )
    }
    report_hashes = {
        "locomotion": "a" * 64,
        "tracking": "b" * 64,
        "onnx": "c" * 64,
    }
    receipt = {
        "schema_version": DEADLINE_FALLBACK_RECEIPT_SCHEMA_VERSION,
        "gate": DEADLINE_POST_CANARY_RECEIPT_GATE,
        "status": "pass",
        "revision": _post_canary_marker()["revision"],
        "checkpoint": {
            "path": "/portable/model_10099.pt",
            "sha256": CANARY_SHA,
            "iteration": 10_099,
            "completed_updates": 10_100,
        },
        "lineage": _marker(),
        "post_canary_authorization": _post_canary_marker(),
        "strict_failure_report": {
            "path": "/portable/strict.json",
            "sha256": CANARY_STRICT_SHA,
            "profile": "whole_body_foot_activation_canary_reachable_safety_v1",
            "status": "fail",
            "failed_checks": ["hand_tracking_rms"],
        },
        "fallback_tracking_report": {
            "path": "/portable/fallback.json",
            "sha256": report_hashes["tracking"],
            "profile": DEADLINE_CANARY_FALLBACK_PROFILE,
            "status": "pass",
            "checks": checks,
        },
        "full_stage_gate": {
            "path": "/portable/gate.json",
            "sha256": "d" * 64,
            "schema_version": 2,
            "status": "pass",
            "checkpoint_sha256": CANARY_SHA,
            "tracking_profile": DEADLINE_CANARY_FALLBACK_PROFILE,
            "report_sha256": report_hashes,
        },
        "promotion": {
            "eligible": True,
            "completed_updates": 10_100,
            "next_completed_updates": 15_000,
            "only_threshold_change": "hand_rms_m_max_0.030_to_0.035",
            "hand_p95_m_max": 0.05,
            "foot_activation_checks_changed": False,
            "safety_thresholds_changed": False,
            "locomotion_gate_changed": False,
            "onnx_gate_changed": False,
            "requires_schema_v2_full_stage_gate": True,
        },
    }
    assert validate_deadline_post_canary_receipt_payload(
        receipt,
        checkpoint_sha256=CANARY_SHA,
        receipt_sha256=MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_RECEIPT_SHA256,
    ) == dict(receipt)
    with pytest.raises(ValueError, match="receipt SHA-256"):
        validate_deadline_post_canary_receipt_payload(
            receipt,
            checkpoint_sha256=CANARY_SHA,
            receipt_sha256="0" * 64,
        )
    drifted = copy.deepcopy(receipt)
    drifted["strict_failure_report"]["sha256"] = STRICT_SHA
    with pytest.raises(ValueError, match="strict evidence"):
        validate_deadline_post_canary_receipt_payload(
            drifted,
            checkpoint_sha256=CANARY_SHA,
            receipt_sha256=MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_RECEIPT_SHA256,
        )
    with pytest.raises(ValueError, match="identity drifted"):
        validate_deadline_post_canary_receipt_payload(
            receipt,
            checkpoint_sha256=SELECTED_SHA,
            receipt_sha256=MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_RECEIPT_SHA256,
        )
    drifted = copy.deepcopy(receipt)
    drifted["fallback_tracking_report"]["checks"]["hand_tracking_p95"] = False
    with pytest.raises(ValueError, match="fallback evidence"):
        validate_deadline_post_canary_receipt_payload(
            drifted,
            checkpoint_sha256=CANARY_SHA,
            receipt_sha256=MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_RECEIPT_SHA256,
        )


def test_validate_receipt_cli_rebuilds_10000_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import mjlab_microban.scripts.teleop_v12_deadline_fallback as fallback

    receipt = tmp_path / "receipt.json"
    expected = {
        "gate": fallback.DEADLINE_FALLBACK_RECEIPT_GATE,
        "checkpoint": {"iteration": 9_999, "completed_updates": 10_000},
    }
    receipt.write_text(json.dumps(expected), encoding="utf-8")
    monkeypatch.setattr(fallback, "build_receipt", lambda **_kwargs: expected)

    assert fallback.main(
        [
            "validate-receipt",
            str(receipt),
            "model_9999.pt",
            "strict.json",
            "fallback.json",
            "gate.json",
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out) == expected


def test_canary_is_exact_one_hop_and_bound_to_selected_parent() -> None:
    payload = _canary_payload(
        checkpoint_path="repo://parent.pt",
        gate_path="repo://gate.json",
        gate_sha="a" * 64,
    )
    assert (
        validate_deadline_fallback_canary_payload(payload, verify_parent_files=False)
        == _marker()
    )
    drifted = copy.deepcopy(payload)
    drifted["iter"] -= 1
    with pytest.raises(ValueError, match="model_10099"):
        validate_deadline_fallback_canary_payload(drifted, verify_parent_files=False)
    drifted = copy.deepcopy(payload)
    drifted["infos"][MICROBAN_TELEOP_V12_DEADLINE_RESUME_SOURCE_INFO_KEY] = (
        deadline_fallback_resume_source(
            checkpoint_path="repo://parent.pt",
            gate_path="repo://gate.json",
            gate_sha256="a" * 64,
            parent_checkpoint_sha256="9" * 64,
        )
    )
    with pytest.raises(ValueError, match="not the selected parent"):
        validate_deadline_fallback_canary_payload(drifted, verify_parent_files=False)
    drifted = copy.deepcopy(payload)
    drifted["infos"][MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY] = (
        corner_rescue_marker(
            parent_checkpoint_sha256="7" * 64,
            parent_strict_tracking_report_sha256="6" * 64,
        )
    )
    with pytest.raises(ValueError, match="corner marker drifted"):
        validate_deadline_fallback_canary_payload(drifted, verify_parent_files=False)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        validate_deadline_fallback_resume_payload(payload, checkpoint_sha256="A" * 64)


def test_parent_gate_hash_binding_and_toctou(monkeypatch, tmp_path: Path) -> None:
    import mjlab_microban.tasks.microban_teleop_v12_deadline_fallback as fallback

    parent = tmp_path / "model_9999.pt"
    gate = tmp_path / "gate.json"
    parent.write_bytes(b"parent")
    gate.write_bytes(b"gate-v1")
    real_sha256 = fallback.sha256_file
    gate_sha = real_sha256(gate)
    payload = _canary_payload(
        checkpoint_path=str(parent), gate_path=str(gate), gate_sha=gate_sha
    )
    source_payload = _source_payload()

    def digest(path: Path) -> str:
        if Path(path) == parent:
            return SELECTED_SHA
        return real_sha256(Path(path))

    monkeypatch.setattr(fallback, "sha256_file", digest)
    monkeypatch.setattr(
        fallback.torch, "load", lambda *_args, **_kwargs: source_payload
    )
    validate_deadline_fallback_canary_payload(payload, verify_parent_files=True)

    def mutate_gate(*_args, **_kwargs):
        gate.write_bytes(b"gate-changed-during-validation")
        return source_payload

    monkeypatch.setattr(fallback.torch, "load", mutate_gate)
    with pytest.raises(ValueError, match="immediate-parent files changed"):
        validate_deadline_fallback_canary_payload(payload, verify_parent_files=True)


def test_training_and_save_endpoints_are_exact() -> None:
    validate_deadline_fallback_training_request(
        current_iteration=10_000,
        common_step_counter=240_000,
        num_learning_iterations=100,
        save_interval=15_000,
    )
    for change in (
        {"current_iteration": 10_001},
        {"common_step_counter": 240_024},
        {"num_learning_iterations": 99},
        {"save_interval": 100},
    ):
        values = {
            "current_iteration": 10_000,
            "common_step_counter": 240_000,
            "num_learning_iterations": 100,
            "save_interval": 15_000,
            **change,
        }
        with pytest.raises(ValueError, match="10000->10100"):
            validate_deadline_fallback_training_request(**values)
    validate_deadline_fallback_save_endpoint(
        iteration=10_099,
        common_step_counter=242_400,
        filename="model_10099.pt",
    )
    validate_deadline_fallback_training_request(
        current_iteration=10_100,
        common_step_counter=242_400,
        num_learning_iterations=4_900,
        save_interval=15_000,
    )
    validate_deadline_fallback_save_endpoint(
        iteration=14_999,
        common_step_counter=360_000,
        filename="model_14999.pt",
    )
    with pytest.raises(RuntimeError, match="model10099"):
        validate_deadline_fallback_save_endpoint(
            iteration=10_000,
            common_step_counter=240_024,
            filename="model_10000.pt",
        )


def test_shell_launchers_route_through_explicit_deadline_mode() -> None:
    subprocess.run(
        ["bash", "-n", str(EVALUATOR), str(CANARY_EVALUATOR), str(TRAINER)],
        check=True,
    )
    evaluator = EVALUATOR.read_text(encoding="utf-8")
    trainer = TRAINER.read_text(encoding="utf-8")
    canary_evaluator = CANARY_EVALUATOR.read_text(encoding="utf-8")
    for required in (
        "create-deadline-fallback",
        "--deadline-fallback",
        "SELECTED_SHA=",
        "realpath --",
        "create-receipt",
        "validate-receipt",
    ):
        assert required in evaluator
    for required in (
        "resume-mode",
        "--agent.deadline-fallback-resume True",
        "save_interval=15000",
        "deadline_fallback_canary_complete",
        "deadline_fallback_post_canary",
    ):
        assert required in trainer
    for required in (
        "create-deadline-canary-fallback",
        "--deadline-canary-fallback",
        "CANARY_SHA=",
        "create-canary-receipt",
        "validate-canary-receipt",
    ):
        assert required in canary_evaluator
