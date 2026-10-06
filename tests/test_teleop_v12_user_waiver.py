"""The provisional user waiver accepts exactly its three recorded exceptions."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from test_teleop_v12_deployment import (
    _bootstrap,
    _evidence,
    _microban_identity,
)
from test_teleop_v12_stage import _onnx_report, _tracking_report

from mjlab_microban.scripts import export_teleop_v12_deployment as deployment
from mjlab_microban.scripts import teleop_v12_user_waiver as waiver
from mjlab_microban.scripts.evaluate_teleop_v12_tracking import (
    FINAL_COMPLETION_ALLOWANCE_PROFILE,
    _acceptance,
)
from mjlab_microban.scripts.teleop_v12_onnx_gate import (
    ONNX_PARITY_RELATIVE_TOLERANCE,
    ONNX_PARITY_RULE,
    ONNX_REFERENCE_PARITY_CHECK,
    ONNX_RUNTIME_CPU_PARITY_CHECK,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
)

IDENTITY = {
    "sha256": waiver.USER_WAIVER_CHECKPOINT_SHA256,
    "iteration": 14_999,
    "completed_updates": 15_000,
}
ATOL = 2.0e-5


def _with_direction(
    report: dict, scenario: str, axis: str, signed: float
) -> dict:
    """Set one scenario axis's signed response and recompute every check."""

    report = deepcopy(report)
    result = next(item for item in report["results"] if item["name"] == scenario)
    command = result["directional_response"][axis]["command"]
    mean = signed if command > 0.0 else -signed
    item = result["directional_response"][axis]
    item.update(
        measured_mean=mean,
        signed_response=signed,
        passed=signed >= item["minimum_signed_response"],
    )
    result["measured_velocity_body"][axis]["mean"] = mean
    result["twist_directional_response_passed"] = all(
        value["passed"] for value in result["directional_response"].values()
    )
    return _rescore(report)


def _rescore(report: dict) -> dict:
    checks, status = _acceptance(report["results"], report["profile"])
    report["checks"] = checks
    report["status"] = status
    return report


def _waived_tracking() -> dict:
    base = _tracking_report(dict(IDENTITY), profile=FINAL_COMPLETION_ALLOWANCE_PROFILE)
    assert base["status"] == "pass"
    return _with_direction(
        base, "mixed_forward_left", "vy_m_s", waiver.USER_WAIVER_TWIST_MEASURED
    )


def test_tracking_waiver_accepts_only_the_recorded_lateral_miss() -> None:
    report = _waived_tracking()
    assert report["status"] == "fail"
    assert {name for name, ok in report["checks"].items() if not ok} == {
        "twist_directional_response"
    }
    waiver.validate_user_waiver_tracking_report(report, dict(IDENTITY))


@pytest.mark.parametrize(
    "case",
    [
        "other_checkpoint",
        "other_scenario",
        "other_axis_vx",
        "other_axis_yaw",
        "other_value",
        "second_scenario_also_fails",
        "passing_report",
        "hand_accuracy_also_fails",
        "soft_limit_also_fails",
        "fall_also",
    ],
)
def test_tracking_waiver_refuses_anything_else(case: str) -> None:
    report = _waived_tracking()
    identity = dict(IDENTITY)
    if case == "other_checkpoint":
        identity["sha256"] = "0" * 64
        report["checkpoint"] = dict(identity)
    elif case == "other_scenario":
        report = _tracking_report(
            dict(IDENTITY), profile=FINAL_COMPLETION_ALLOWANCE_PROFILE
        )
        report = _with_direction(
            report, "mixed_backward_right", "vy_m_s", waiver.USER_WAIVER_TWIST_MEASURED
        )
    elif case == "other_axis_vx":
        report = _tracking_report(
            dict(IDENTITY), profile=FINAL_COMPLETION_ALLOWANCE_PROFILE
        )
        report = _with_direction(report, "mixed_forward_left", "vx_m_s", 0.01)
    elif case == "other_axis_yaw":
        report = _with_direction(report, "mixed_forward_left", "yaw_rad_s", 0.1)
    elif case == "other_value":
        report = _with_direction(report, "mixed_forward_left", "vy_m_s", -0.05)
    elif case == "second_scenario_also_fails":
        report = _with_direction(report, "mixed_backward_right", "vy_m_s", 0.0)
    elif case == "passing_report":
        report = _tracking_report(
            dict(IDENTITY), profile=FINAL_COMPLETION_ALLOWANCE_PROFILE
        )
    elif case == "hand_accuracy_also_fails":
        result = next(r for r in report["results"] if r["name"] == "max_hands_left")
        result["target_error"]["active_hand"].update(rms=0.05, max=0.06)
        report = _rescore(report)
        assert not report["checks"]["hand_tracking_rms"]
    elif case == "soft_limit_also_fails":
        report["results"][0]["maximum_actual_soft_limit_violation_rad"] = 0.1
        report = _rescore(report)
    elif case == "fall_also":
        report["results"][1]["fell"] = True
        report = _rescore(report)
    with pytest.raises(ValueError):
        waiver.validate_user_waiver_tracking_report(report, identity)


def _waived_onnx(tmp_path: Path) -> dict:
    onnx_path = tmp_path / "gate.onnx"
    onnx_path.write_bytes(b"onnx")
    report = _onnx_report(dict(IDENTITY), onnx_path)
    magnitudes = [10.0 + index for index in range(64)]
    magnitudes[waiver.USER_WAIVER_ONNX_SAMPLE_INDEX] = (
        waiver.USER_WAIVER_ONNX_SAMPLE_MAXIMUM_EXPECTED_OUTPUT
    )
    magnitudes[54] = waiver.USER_WAIVER_MAXIMUM_EXPECTED_OUTPUT
    runtime_ratios = [0.5] * 64
    runtime_ratios[waiver.USER_WAIVER_ONNX_SAMPLE_INDEX] = (
        waiver.USER_WAIVER_ONNX_BOUND_RATIO
    )
    reference_ratios = [0.25] * 64

    def errors(ratios: list[float]) -> list[float]:
        return [
            ratio * (ATOL + ONNX_PARITY_RELATIVE_TOLERANCE * magnitude)
            for ratio, magnitude in zip(ratios, magnitudes, strict=True)
        ]

    runtime_errors = errors(runtime_ratios)
    runtime_errors[waiver.USER_WAIVER_ONNX_SAMPLE_INDEX] = (
        waiver.USER_WAIVER_ONNX_ABSOLUTE_ERROR
    )
    reference_errors = errors(reference_ratios)
    report["status"] = "fail"
    report["parity_failure_recording"] = True
    report["failed_checks"] = [ONNX_RUNTIME_CPU_PARITY_CHECK]
    report["onnx"].update(
        {
            "relative_tolerance": ONNX_PARITY_RELATIVE_TOLERANCE,
            "parity_rule": ONNX_PARITY_RULE,
            "maximum_absolute_expected_output": max(magnitudes),
            "reference_evaluator_maximum_bound_ratio": max(reference_ratios),
            "onnxruntime_cpu_maximum_bound_ratio": max(runtime_ratios),
            "reference_evaluator_maximum_absolute_error": max(reference_errors),
            "onnxruntime_cpu_maximum_absolute_error": max(runtime_errors),
            "per_sample": {
                "reference_evaluator_absolute_errors": reference_errors,
                "onnxruntime_cpu_absolute_errors": runtime_errors,
                "reference_evaluator_bound_ratios": reference_ratios,
                "onnxruntime_cpu_bound_ratios": runtime_ratios,
                "maximum_absolute_expected_outputs": magnitudes,
            },
        }
    )
    return report


def _resummarize(report: dict) -> dict:
    onnx = report["onnx"]
    per = onnx["per_sample"]
    onnx["reference_evaluator_maximum_bound_ratio"] = max(
        per["reference_evaluator_bound_ratios"]
    )
    onnx["onnxruntime_cpu_maximum_bound_ratio"] = max(per["onnxruntime_cpu_bound_ratios"])
    onnx["onnxruntime_cpu_maximum_absolute_error"] = max(
        per["onnxruntime_cpu_absolute_errors"]
    )
    onnx["reference_evaluator_maximum_absolute_error"] = max(
        per["reference_evaluator_absolute_errors"]
    )
    onnx["maximum_absolute_expected_output"] = max(
        per["maximum_absolute_expected_outputs"]
    )
    return report


def _set_sample(report: dict, index: int, *, ratio: float, which: str = "onnxruntime_cpu") -> dict:
    per = report["onnx"]["per_sample"]
    magnitude = per["maximum_absolute_expected_outputs"][index]
    prefix = "onnxruntime_cpu" if which == "onnxruntime_cpu" else "reference_evaluator"
    per[f"{prefix}_bound_ratios"][index] = ratio
    per[f"{prefix}_absolute_errors"][index] = ratio * (
        ATOL + ONNX_PARITY_RELATIVE_TOLERANCE * magnitude
    )
    return _resummarize(report)


def test_onnx_waiver_accepts_only_the_recorded_sample_and_maximum(
    tmp_path: Path,
) -> None:
    path, sha = waiver.validate_user_waiver_onnx_report(
        _waived_onnx(tmp_path), dict(IDENTITY)
    )
    assert path == (tmp_path / "gate.onnx").resolve()
    assert len(sha) == 64


@pytest.mark.parametrize(
    "case",
    [
        "other_checkpoint",
        "status_pass",
        "other_failed_check",
        "second_sample_over",
        "worse_waived_ratio",
        "other_sample_index",
        "reference_over",
        "other_corpus_maximum",
        "maximum_inconsistent",
        "neutral_parity_failed",
        "ratios_inconsistent",
        "not_recorded",
    ],
)
def test_onnx_waiver_refuses_anything_else(tmp_path: Path, case: str) -> None:
    report = _waived_onnx(tmp_path)
    identity = dict(IDENTITY)
    sample = waiver.USER_WAIVER_ONNX_SAMPLE_INDEX
    if case == "other_checkpoint":
        identity["sha256"] = "0" * 64
        report["checkpoint"] = dict(identity)
    elif case == "status_pass":
        report["status"] = "pass"
    elif case == "other_failed_check":
        report["failed_checks"] = [ONNX_REFERENCE_PARITY_CHECK]
    elif case == "second_sample_over":
        report = _set_sample(report, 3, ratio=1.001)
    elif case == "worse_waived_ratio":
        report = _set_sample(report, sample, ratio=1.05)
    elif case == "other_sample_index":
        per = report["onnx"]["per_sample"]
        for name in per:
            per[name][sample], per[name][sample - 1] = (
                per[name][sample - 1],
                per[name][sample],
            )
        report = _resummarize(report)
    elif case == "reference_over":
        report = _set_sample(report, 5, ratio=1.2, which="reference")
        report["failed_checks"] = [
            ONNX_REFERENCE_PARITY_CHECK,
            ONNX_RUNTIME_CPU_PARITY_CHECK,
        ]
    elif case == "other_corpus_maximum":
        report["onnx"]["per_sample"]["maximum_absolute_expected_outputs"][54] = 230.0
        report = _set_sample(report, 54, ratio=0.5)
    elif case == "maximum_inconsistent":
        report["onnx"]["onnxruntime_cpu_maximum_bound_ratio"] = 0.9
    elif case == "neutral_parity_failed":
        report["neutral_legacy_parity"]["maximum_absolute_error"] = 1.0e-3
    elif case == "ratios_inconsistent":
        report["onnx"]["per_sample"]["onnxruntime_cpu_absolute_errors"][7] *= 2.0
        report = _resummarize(report)
    elif case == "not_recorded":
        report.pop("parity_failure_recording")
    with pytest.raises(ValueError):
        waiver.validate_user_waiver_onnx_report(report, identity)


def test_record_names_the_three_exceptions_and_both_user_decisions() -> None:
    record = waiver.user_waiver_record()
    assert record["stage_gate_pass"] is False and record["provisional"] is True
    assert record["checkpoint"]["sha256"] == waiver.USER_WAIVER_CHECKPOINT_SHA256
    assert [item["id"] for item in record["waived"]] == [
        "tracking_twist_mixed_forward_left_lateral",
        "onnxruntime_cpu_random_parity_sample_60",
        "random_parity_expected_output_above_robot_cap",
    ]
    twist, ratio, cap = record["waived"]
    assert (twist["scenario"], twist["axis"]) == ("mixed_forward_left", "vy_m_s")
    assert twist["measured_signed_response"] == -0.016997758105397224
    assert twist["minimum_signed_response"] == 0.02
    assert (ratio["sample_index"], ratio["bound_ratio"]) == (60, 1.0178567171096802)
    assert cap["maximum_absolute_expected_output"] == 219.99908447265625
    assert cap["robot_cap"] == 200.0
    assert [item["verbatim"] for item in record["user_decisions"]] == [
        "苦手な動きが1つ残ったまま実機の前傾版ブランチに入れてよい。その後直す、が良いと思います",
        "「A（許容して入れる）」",
    ]
    covered = {key for item in record["user_decisions"] for key in item["covers"]}
    assert covered == {item["id"] for item in record["waived"]}
    assert record["tracking_profile"] == (
        "full_body_reachable_performance_perturbation_v2_completion_allowance_v1_"
        "user_waiver_mixed_forward_left_twist_v1"
    )
    # The record round-trips through the package's ASCII JSON unchanged.
    assert json.loads(deployment._json(record)) == record


def _waiver_gate(gate: dict) -> dict:
    gate = deepcopy(gate)
    gate.update(
        status=waiver.USER_WAIVER_GATE_STATUS,
        checkpoint_sha256=waiver.USER_WAIVER_CHECKPOINT_SHA256,
        tracking_profile=waiver.USER_WAIVER_TRACKING_PROFILE,
        user_waiver=waiver.user_waiver_record(),
        user_waiver_sha256=waiver.canonical_json_sha256(waiver.user_waiver_record()),
    )
    return gate


@pytest.mark.parametrize(
    ("case", "accepted"),
    [
        ("exact", True),
        ("other_checkpoint", False),
        ("record_drift", False),
        ("record_sha_drift", False),
        ("profile_drift", False),
        ("status_pass_with_record", False),
        ("ordinary_gate_with_waiver_profile", False),
    ],
)
def test_packager_final_gate_accepts_only_the_exact_waiver(
    tmp_path: Path, case: str, accepted: bool
) -> None:
    checkpoint = tmp_path / "model_14999.pt"
    gate = _waiver_gate(_evidence(tmp_path)[0])
    sha = waiver.USER_WAIVER_CHECKPOINT_SHA256
    if case == "other_checkpoint":
        sha = "9" * 64
        gate["checkpoint_sha256"] = sha
    elif case == "record_drift":
        gate["user_waiver"]["waived"][0]["scenario"] = "mixed_backward_right"
    elif case == "record_sha_drift":
        gate["user_waiver_sha256"] = "0" * 64
    elif case == "profile_drift":
        gate["tracking_profile"] = FINAL_COMPLETION_ALLOWANCE_PROFILE
    elif case == "status_pass_with_record":
        gate["status"] = "pass"
    elif case == "ordinary_gate_with_waiver_profile":
        gate = _evidence(tmp_path)[0]
        gate["checkpoint_sha256"] = sha
        gate["tracking_profile"] = waiver.USER_WAIVER_TRACKING_PROFILE
    if accepted:
        deployment._require_final_gate(gate, checkpoint=checkpoint, checkpoint_sha256=sha)
    else:
        with pytest.raises(ValueError):
            deployment._require_final_gate(
                gate, checkpoint=checkpoint, checkpoint_sha256=sha
            )


def _rule_evidence(**overrides: object) -> dict:
    evidence = {
        "parity_rule": ONNX_PARITY_RULE,
        "relative_tolerance": ONNX_PARITY_RELATIVE_TOLERANCE,
        "maximum_absolute_expected_output": waiver.USER_WAIVER_MAXIMUM_EXPECTED_OUTPUT,
        "reference_evaluator_maximum_bound_ratio": 0.2558188736438751,
        "onnxruntime_cpu_maximum_bound_ratio": waiver.USER_WAIVER_ONNX_BOUND_RATIO,
    }
    evidence.update(overrides)
    return evidence


def test_parity_rule_metadata_ships_only_the_waived_values() -> None:
    metadata = deployment._onnx_parity_rule_metadata(_rule_evidence(), user_waiver=True)
    assert metadata["v12_onnxruntime_cpu_max_bound_ratio"] == "1.0178567171096802"
    assert metadata["v12_onnx_parity_max_abs_expected_output"] == "219.99908447265625"
    with pytest.raises(ValueError):  # not without the waiver
        deployment._onnx_parity_rule_metadata(_rule_evidence())
    for drift in (
        {"onnxruntime_cpu_maximum_bound_ratio": 1.03},
        {"onnxruntime_cpu_maximum_bound_ratio": 0.5},
        {"maximum_absolute_expected_output": 250.0},
        {"reference_evaluator_maximum_bound_ratio": 1.01},
    ):
        with pytest.raises(ValueError):
            deployment._onnx_parity_rule_metadata(
                _rule_evidence(**drift), user_waiver=True
            )


def _recorded_parity(corpus_sha256: str) -> dict:
    return {
        "semantics": waiver.RECORDED_CORPUS_PARITY_SEMANTICS,
        "corpus_sha256": corpus_sha256,
        "samples": 16,
        "rule": ONNX_PARITY_RULE,
        "atol": ATOL,
        "rtol": ONNX_PARITY_RELATIVE_TOLERANCE,
        "maximum_absolute_expected_output": 3.0,
        "reference_evaluator_maximum_absolute_error": 4.8e-7,
        "onnxruntime_cpu_maximum_absolute_error": 9.5e-7,
        "reference_evaluator_maximum_bound_ratio": 0.02,
        "onnxruntime_cpu_maximum_bound_ratio": 0.04,
        "status": "pass",
    }


def test_package_metadata_names_the_waiver_and_its_recorded_corpus_parity(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model_14999.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_path = tmp_path / "gate.json"
    gate_path.write_text("{}", encoding="utf-8")
    gate, locomotion, tracking, onnx_report, infos = _evidence(tmp_path)
    infos["microban_teleop_recipe_revision"] = (
        MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
    )
    onnx_report["onnx"].update(_rule_evidence())
    corpus_sha = deployment.hashlib.sha256(
        deployment._json(deployment._runtime_smoke_corpus(tracking)).encode("ascii")
    ).hexdigest()
    gate = _waiver_gate(gate)
    gate["user_waiver_recorded_corpus_parity"] = _recorded_parity(corpus_sha)

    def build(gate_value: dict, parity: dict | None) -> dict:
        return deployment.build_v12_deployment_metadata(
            checkpoint=checkpoint,
            checkpoint_sha256=waiver.USER_WAIVER_CHECKPOINT_SHA256,
            gate_path=gate_path,
            gate=gate_value,
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
            user_waiver_recorded_corpus_parity=parity,
        )

    metadata = build(gate, _recorded_parity(corpus_sha))
    assert metadata["v12_stage_gate_status"] == "provisional_user_waiver"
    assert metadata["v12_tracking_profile"] == waiver.USER_WAIVER_TRACKING_PROFILE
    assert metadata["v12_provisional_install"] == "true"
    assert json.loads(metadata["v12_user_waiver_json"]) == waiver.user_waiver_record()
    assert metadata["v12_user_waiver_sha256"] == waiver.canonical_json_sha256(
        waiver.user_waiver_record()
    )
    assert json.loads(
        metadata["v12_user_waiver_recorded_corpus_parity_json"]
    ) == _recorded_parity(corpus_sha)
    for bad in (
        None,
        _recorded_parity("0" * 64),
        {**_recorded_parity(corpus_sha), "onnxruntime_cpu_maximum_bound_ratio": 0.05},
    ):
        with pytest.raises(ValueError):
            build(gate, bad)
