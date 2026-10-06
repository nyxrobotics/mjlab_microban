"""Provisional user-waiver gate for exactly one forward-lean v12 final checkpoint.

TEMPORARY.  Delete this module, tests/test_teleop_v12_user_waiver.py and the
lines tagged ``TEMPORARY user waiver`` in export_teleop_v12_deployment.py when
the twist-ratio replacement model is installed on forward-lean-home.

This is not a stage gate pass.  The user decided (2026-10-06) to install the
forward-lean pose-release ``model_14999.pt`` of run
``2026-10-06_01-05-06_lean_v12_pr_10100_to15000`` on the robot's forward-lean
branch although it fails three named checks, and to replace it with a fixed
model later.  This module encodes that decision and nothing wider:

* tracking: ``twist_directional_response`` fails only in scenario
  ``mixed_forward_left``, only on the lateral axis ``vy_m_s``, with exactly the
  recorded value (-0.0170 m/s against the 0.02 m/s minimum); every other
  tracking check of the 15000 completion-allowance profile passes;
* ONNX: ONNX Runtime CPU full-83 parity on the random check set misses the
  norm-wise bound only on sample 60 (bound ratio 1.0179, exactly the recorded
  value); the reference evaluator, every other sample and the neutral legacy
  parity pass;
* ONNX: the random check set's largest expected output (219.999) is above the
  robot's 200 cap on that magnitude.

Everything is pinned to the checkpoint's SHA-256 and the recorded values, so
any other checkpoint, failed check, scenario, axis, sample or value is refused.
Locomotion 9x300 must pass unchanged, and the user's condition for the ONNX
exceptions -- torch and ONNX (Runtime CPU and reference) agree under the normal
rule on the package's recorded self-test corpus -- is re-run here.  The gate
written by :func:`create_user_waiver_gate` has ``status ==
"provisional_user_waiver"``, which :func:`teleop_v12_stage.validate_gate` (and
so every resume path) refuses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from mjlab_microban.legacy_velocity_diagnostics import publish_json_atomic
from mjlab_microban.scripts.evaluate_teleop_v12_tracking import (
    DIRECTIONAL_RESPONSE_MINIMUM,
    FINAL_COMPLETION_ALLOWANCE_PROFILE,
    required_tracking_profile,
    tracking_profile_completion_allowance,
)
from mjlab_microban.scripts.teleop_v12_bootstrap_gate import ONNX_PARITY_TOLERANCE
from mjlab_microban.scripts.teleop_v12_onnx_gate import (
    ONNX_PARITY_RELATIVE_TOLERANCE,
    ONNX_PARITY_RULE,
    parity_bound_ratio,
)
from mjlab_microban.scripts.teleop_v12_stage import (
    _checkpoint_identity,
    _load_json,
    _validate_locomotion_report,
    _validate_onnx_report,
    _validate_tracking_report,
    lateral_fidelity_gate_marker,
    pose_release_final_rescue_gate_marker,
)
from mjlab_microban.tasks.microban_teleop_v12_bootstrap import (
    portable_bootstrap_artifact_path,
    resolve_bootstrap_artifact_path,
    sha256_file,
)
from mjlab_microban.tasks.microban_teleop_v12_corner_rescue import (
    MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY,
)
from mjlab_microban.tasks.microban_teleop_v12_deadline_fallback import (
    MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_INFO_KEY,
    MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_INFO_KEY,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
)
from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_lineage import (
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_SWITCH_INFO_KEY,
    validate_hand_pose_release_recipe_switch_marker,
)
from mjlab_microban.tasks.microban_teleop_v12_home_pose import (
    TELEOP_V12_HOME_POSE_INFO_KEY,
)

ONNX_REFERENCE_PARITY_CHECK = "reference_evaluator_full83_parity"
ONNX_RUNTIME_CPU_PARITY_CHECK = "onnxruntime_cpu_full83_parity"
USER_WAIVER_REVISION = "provisional_user_waiver_lean_pr_model_14999_v1"
USER_WAIVER_GATE_STATUS = "provisional_user_waiver"
USER_WAIVER_TRACKING_PROFILE = (
    f"{FINAL_COMPLETION_ALLOWANCE_PROFILE}_user_waiver_mixed_forward_left_twist_v1"
)
USER_WAIVER_RUN = "2026-10-06_01-05-06_lean_v12_pr_10100_to15000"
USER_WAIVER_CHECKPOINT_FILENAME = "model_14999.pt"
USER_WAIVER_CHECKPOINT_ITERATION = 14_999
USER_WAIVER_COMPLETED_UPDATES = 15_000
USER_WAIVER_CHECKPOINT_SHA256 = (
    "b41a6cb2df5e05909f75b35f50889b298559e97bf2ee56d1c35277fb2cc71e54"
)
# (1) tracking: the one waived directional-response miss.
USER_WAIVER_TWIST_CHECK = "twist_directional_response"
USER_WAIVER_TWIST_SCENARIO = "mixed_forward_left"
USER_WAIVER_TWIST_AXIS = "vy_m_s"
USER_WAIVER_TWIST_COMMAND = 0.3
USER_WAIVER_TWIST_MEASURED = -0.016997758105397224
# (2) ONNX Runtime CPU full-83 random-corpus parity: the one waived sample.
USER_WAIVER_ONNX_RANDOM_CORPUS = (
    "torch_randn_seed20260925_skip_10000x83_then_64x83_all_columns"
)
USER_WAIVER_ONNX_SAMPLE_INDEX = 60
USER_WAIVER_ONNX_BOUND_RATIO = 1.0178567171096802
USER_WAIVER_ONNX_ABSOLUTE_ERROR = 9.1552734375e-05
USER_WAIVER_ONNX_SAMPLE_MAXIMUM_EXPECTED_OUTPUT = 69.94658660888672
# (3) the random corpus's largest expected output vs the robot's cap on it.
USER_WAIVER_MAXIMUM_EXPECTED_OUTPUT = 219.99908447265625
ROBOT_V12_ONNX_PARITY_MAX_EXPECTED_OUTPUT = 200.0
USER_WAIVER_DECISIONS = (
    {
        "date": "2026-10-06",
        "time": "about 10:00 JST",
        "verbatim": (
            "苦手な動きが1つ残ったまま実機の前傾版ブランチに入れてよい。"
            "その後直す、が良いと思います"
        ),
        "covers": ["tracking_twist_mixed_forward_left_lateral"],
    },
    {
        "date": "2026-10-06",
        "time": "about 10:20 JST",
        "verbatim": "「A（許容して入れる）」",
        "covers": [
            "onnxruntime_cpu_random_parity_sample_60",
            "random_parity_expected_output_above_robot_cap",
        ],
        "condition": (
            "torch, ONNX Runtime CPU and the ONNX reference evaluator agree "
            "under the normal norm-wise rule (bound ratio <= 1) on the "
            "package's recorded self-test corpus"
        ),
    },
)
USER_WAIVER_KNOWN_LIMITATION = (
    "stick forward-left plus left yaw while reaching with the hands (evaluator "
    "scenario mixed_forward_left, with its periodic push): the robot moves "
    "almost not at all to the left, lateral response -0.0170 m/s against the "
    "0.02 m/s minimum; a fixed model will replace this provisional one"
)
RECORDED_CORPUS_PARITY_SEMANTICS = (
    "torch_vs_onnxruntime_cpu_and_reference_on_final_tracking_runtime_smoke_"
    "observations_normwise_rule_v1"
)


def user_waiver_record() -> dict[str, Any]:
    """The exact waiver record a gate and its package carry (robot-pinned)."""

    return {
        "schema_version": 1,
        "revision": USER_WAIVER_REVISION,
        "install_kind": "provisional_user_waiver",
        "stage_gate_pass": False,
        "provisional": True,
        "to_be_replaced_by_a_fixed_model": True,
        "checkpoint": {
            "run": USER_WAIVER_RUN,
            "filename": USER_WAIVER_CHECKPOINT_FILENAME,
            "iteration": USER_WAIVER_CHECKPOINT_ITERATION,
            "completed_updates": USER_WAIVER_COMPLETED_UPDATES,
            "sha256": USER_WAIVER_CHECKPOINT_SHA256,
        },
        "recipe_revision": MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
        "base_tracking_profile": FINAL_COMPLETION_ALLOWANCE_PROFILE,
        "tracking_profile": USER_WAIVER_TRACKING_PROFILE,
        "waived": [
            {
                "id": "tracking_twist_mixed_forward_left_lateral",
                "report": "tracking",
                "check": USER_WAIVER_TWIST_CHECK,
                "scenario": USER_WAIVER_TWIST_SCENARIO,
                "axis": USER_WAIVER_TWIST_AXIS,
                "command": USER_WAIVER_TWIST_COMMAND,
                "measured_signed_response": USER_WAIVER_TWIST_MEASURED,
                "minimum_signed_response": DIRECTIONAL_RESPONSE_MINIMUM[
                    USER_WAIVER_TWIST_AXIS
                ],
            },
            {
                "id": "onnxruntime_cpu_random_parity_sample_60",
                "report": "onnx",
                "check": ONNX_RUNTIME_CPU_PARITY_CHECK,
                "corpus": USER_WAIVER_ONNX_RANDOM_CORPUS,
                "rule": ONNX_PARITY_RULE,
                "atol": ONNX_PARITY_TOLERANCE,
                "rtol": ONNX_PARITY_RELATIVE_TOLERANCE,
                "sample_index": USER_WAIVER_ONNX_SAMPLE_INDEX,
                "absolute_error": USER_WAIVER_ONNX_ABSOLUTE_ERROR,
                "sample_maximum_absolute_expected_output": (
                    USER_WAIVER_ONNX_SAMPLE_MAXIMUM_EXPECTED_OUTPUT
                ),
                "bound_ratio": USER_WAIVER_ONNX_BOUND_RATIO,
                "maximum_allowed_bound_ratio": 1.0,
            },
            {
                "id": "random_parity_expected_output_above_robot_cap",
                "report": "onnx",
                "check": "robot_v12_onnx_parity_max_abs_expected_output",
                "corpus": USER_WAIVER_ONNX_RANDOM_CORPUS,
                "maximum_absolute_expected_output": (
                    USER_WAIVER_MAXIMUM_EXPECTED_OUTPUT
                ),
                "robot_cap": ROBOT_V12_ONNX_PARITY_MAX_EXPECTED_OUTPUT,
            },
        ],
        "user_decisions": deepcopy(list(USER_WAIVER_DECISIONS)),
        "known_limitation": USER_WAIVER_KNOWN_LIMITATION,
        "unwaived": (
            "every other tracking check of the base profile, locomotion 9x300, "
            "neutral legacy parity, ONNX reference parity, every other random "
            "parity sample, the recorded-corpus parity and the robot runtime "
            "checks run and must pass unchanged"
        ),
    }


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def is_user_waiver_gate(gate: Mapping[str, Any]) -> bool:
    return gate.get("status") == USER_WAIVER_GATE_STATUS


def _require_waived_identity(identity: Mapping[str, Any]) -> None:
    if (
        identity.get("sha256") != USER_WAIVER_CHECKPOINT_SHA256
        or identity.get("iteration") != USER_WAIVER_CHECKPOINT_ITERATION
        or identity.get("completed_updates") != USER_WAIVER_COMPLETED_UPDATES
    ):
        raise ValueError("The user waiver covers only its one recorded checkpoint")


def validate_user_waiver_tracking_report(
    report: dict[str, Any], expected_identity: dict[str, int | str]
) -> None:
    """Accept the base-profile report failing only the one waived lateral miss."""

    _require_waived_identity(expected_identity)
    _validate_tracking_report(
        report,
        expected_identity,
        profile_override=FINAL_COMPLETION_ALLOWANCE_PROFILE,
        allowed_failed_checks=frozenset((USER_WAIVER_TWIST_CHECK,)),
    )
    checks = report.get("checks")
    if (
        report.get("status") != "fail"
        or not isinstance(checks, dict)
        or {name for name, passed in checks.items() if passed is not True}
        != {USER_WAIVER_TWIST_CHECK}
    ):
        raise ValueError(
            "The user waiver requires a report failing only "
            f"{USER_WAIVER_TWIST_CHECK}"
        )
    failed: list[tuple[str, str, dict[str, Any]]] = []
    for result in report["results"]:
        for axis, item in result["directional_response"].items():
            if item.get("passed") is not True:
                failed.append((result["name"], axis, item))
    if len(failed) != 1:
        raise ValueError("The user waiver covers exactly one failed directional axis")
    scenario, axis, item = failed[0]
    if (
        scenario != USER_WAIVER_TWIST_SCENARIO
        or axis != USER_WAIVER_TWIST_AXIS
        or item.get("command") != USER_WAIVER_TWIST_COMMAND
        or item.get("signed_response") != USER_WAIVER_TWIST_MEASURED
        or item.get("minimum_signed_response")
        != DIRECTIONAL_RESPONSE_MINIMUM[USER_WAIVER_TWIST_AXIS]
    ):
        raise ValueError(
            "The user waiver covers only the recorded mixed_forward_left lateral "
            f"miss, not {scenario}/{axis}={item.get('signed_response')!r}"
        )


def _finite_list(value: object, *, count: int, label: str) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != count
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) < 0.0
            for item in value
        )
    ):
        raise ValueError(f"ONNX per-sample evidence {label} is malformed")
    return [float(item) for item in value]


def validate_user_waiver_onnx_report(
    report: dict[str, Any], expected_identity: dict[str, int | str]
) -> tuple[Path, str]:
    """Accept the recorded ONNX evidence failing only the two waived items.

    Every field the stage gate checks is checked unchanged by
    ``_validate_onnx_report`` on a copy whose runtime ratio is the maximum over
    the samples other than the waived one; the waived sample and the corpus
    maximum must be exactly the recorded values.
    """

    _require_waived_identity(expected_identity)
    onnx = report.get("onnx")
    per_sample = onnx.get("per_sample") if isinstance(onnx, dict) else None
    if (
        report.get("status") != "fail"
        or report.get("parity_failure_recording") is not True
        or report.get("failed_checks") != [ONNX_RUNTIME_CPU_PARITY_CHECK]
        or not isinstance(per_sample, dict)
        or set(per_sample)
        != {
            "reference_evaluator_absolute_errors",
            "onnxruntime_cpu_absolute_errors",
            "reference_evaluator_bound_ratios",
            "onnxruntime_cpu_bound_ratios",
            "maximum_absolute_expected_outputs",
        }
    ):
        raise ValueError(
            "The user waiver requires recorded ONNX evidence failing only "
            f"{ONNX_RUNTIME_CPU_PARITY_CHECK}"
        )
    count = 64
    values = {
        name: _finite_list(per_sample[name], count=count, label=name)
        for name in per_sample
    }
    runtime_ratios = values["onnxruntime_cpu_bound_ratios"]
    reference_ratios = values["reference_evaluator_bound_ratios"]
    magnitudes = values["maximum_absolute_expected_outputs"]
    for errors, ratios in (
        (values["onnxruntime_cpu_absolute_errors"], runtime_ratios),
        (values["reference_evaluator_absolute_errors"], reference_ratios),
    ):
        for error, ratio, magnitude in zip(errors, ratios, magnitudes, strict=True):
            bound = ONNX_PARITY_TOLERANCE + ONNX_PARITY_RELATIVE_TOLERANCE * magnitude
            if not math.isclose(error / bound, ratio, rel_tol=1.0e-5, abs_tol=0.0):
                raise ValueError("ONNX per-sample bound ratios are inconsistent")
    if (
        onnx.get("onnxruntime_cpu_maximum_bound_ratio") != max(runtime_ratios)
        or onnx.get("reference_evaluator_maximum_bound_ratio")
        != max(reference_ratios)
        or onnx.get("onnxruntime_cpu_maximum_absolute_error")
        != max(values["onnxruntime_cpu_absolute_errors"])
        or onnx.get("reference_evaluator_maximum_absolute_error")
        != max(values["reference_evaluator_absolute_errors"])
        or onnx.get("maximum_absolute_expected_output") != max(magnitudes)
    ):
        raise ValueError("ONNX per-sample evidence does not match its maxima")
    over = [index for index, ratio in enumerate(runtime_ratios) if ratio > 1.0]
    sample = USER_WAIVER_ONNX_SAMPLE_INDEX
    if (
        over != [sample]
        or runtime_ratios[sample] != USER_WAIVER_ONNX_BOUND_RATIO
        or values["onnxruntime_cpu_absolute_errors"][sample]
        != USER_WAIVER_ONNX_ABSOLUTE_ERROR
        or magnitudes[sample] != USER_WAIVER_ONNX_SAMPLE_MAXIMUM_EXPECTED_OUTPUT
        or max(reference_ratios) > 1.0
        or max(magnitudes) != USER_WAIVER_MAXIMUM_EXPECTED_OUTPUT
    ):
        raise ValueError(
            "The user waiver covers only the recorded ONNX Runtime CPU parity "
            f"miss on sample {sample} and the recorded corpus maximum"
        )
    normalized = deepcopy(report)
    normalized["status"] = "pass"
    normalized["onnx"]["onnxruntime_cpu_maximum_bound_ratio"] = max(
        ratio for index, ratio in enumerate(runtime_ratios) if index != sample
    )
    return _validate_onnx_report(normalized, expected_identity)


def recorded_corpus_parity(
    actor: Any, onnx_path: Path, tracking: Mapping[str, Any]
) -> dict[str, Any]:
    """Torch vs ONNX on the package's recorded self-test corpus (normal rule).

    ``actor`` is the loaded v12 actor.  Raises unless every sample of both ONNX
    implementations is inside ``atol + rtol * max|expected|``.
    """

    import onnx
    import onnxruntime as ort
    import torch
    from onnx.reference import ReferenceEvaluator

    from mjlab_microban.scripts.export_teleop_v12_deployment import (
        _json,
        _runtime_smoke_corpus,
    )

    corpus = _runtime_smoke_corpus(tracking)
    runtime = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if runtime.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("Recorded-corpus parity did not use CPUExecutionProvider")
    reference = ReferenceEvaluator(onnx.load(onnx_path))
    export_model = actor.as_onnx(verbose=False).cpu().eval()
    stats = {
        "reference_errors": [],
        "runtime_errors": [],
        "reference_ratios": [],
        "runtime_ratios": [],
        "magnitudes": [],
    }
    with torch.inference_mode():
        for row in corpus:
            batch = torch.tensor([row], dtype=torch.float32)
            expected = export_model(batch).detach().cpu().numpy()
            (reference_actual,) = reference.run(None, {"obs": batch.numpy()})
            (runtime_actual,) = runtime.run(None, {"obs": batch.numpy()})
            stats["reference_errors"].append(
                float(np.max(np.abs(reference_actual - expected)))
            )
            stats["runtime_errors"].append(
                float(np.max(np.abs(runtime_actual - expected)))
            )
            stats["reference_ratios"].append(
                parity_bound_ratio(
                    reference_actual, expected, atol=ONNX_PARITY_TOLERANCE
                )
            )
            stats["runtime_ratios"].append(
                parity_bound_ratio(runtime_actual, expected, atol=ONNX_PARITY_TOLERANCE)
            )
            stats["magnitudes"].append(float(np.max(np.abs(expected))))
    result = {
        "semantics": RECORDED_CORPUS_PARITY_SEMANTICS,
        "corpus_sha256": hashlib.sha256(_json(corpus).encode("ascii")).hexdigest(),
        "samples": len(corpus),
        "rule": ONNX_PARITY_RULE,
        "atol": ONNX_PARITY_TOLERANCE,
        "rtol": ONNX_PARITY_RELATIVE_TOLERANCE,
        "maximum_absolute_expected_output": max(stats["magnitudes"]),
        "reference_evaluator_maximum_absolute_error": max(stats["reference_errors"]),
        "onnxruntime_cpu_maximum_absolute_error": max(stats["runtime_errors"]),
        "reference_evaluator_maximum_bound_ratio": max(stats["reference_ratios"]),
        "onnxruntime_cpu_maximum_bound_ratio": max(stats["runtime_ratios"]),
        "status": "pass",
    }
    if not all(
        math.isfinite(float(value))
        for value in (
            result["maximum_absolute_expected_output"],
            result["reference_evaluator_maximum_bound_ratio"],
            result["onnxruntime_cpu_maximum_bound_ratio"],
        )
    ) or max(
        result["reference_evaluator_maximum_bound_ratio"],
        result["onnxruntime_cpu_maximum_bound_ratio"],
    ) > 1.0:
        raise ValueError(
            "Recorded self-test corpus parity failed under the normal rule; the "
            f"user's condition for the ONNX waiver is not met: {result}"
        )
    return result


def _require_waivable_checkpoint(
    checkpoint: Path, iteration: int, completed: int, sha: str, infos: dict
) -> None:
    if (
        sha != USER_WAIVER_CHECKPOINT_SHA256
        or iteration != USER_WAIVER_CHECKPOINT_ITERATION
        or completed != USER_WAIVER_COMPLETED_UPDATES
        or checkpoint.name != USER_WAIVER_CHECKPOINT_FILENAME
        or checkpoint.parent.name != USER_WAIVER_RUN
    ):
        raise ValueError("The user waiver covers only its one recorded checkpoint")
    if (
        infos.get("microban_teleop_recipe_revision")
        != MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
        or required_tracking_profile(
            completed, recipe_revision=infos.get("microban_teleop_recipe_revision")
        )
        != FINAL_COMPLETION_ALLOWANCE_PROFILE
        or infos.get(MICROBAN_TELEOP_V12_DEADLINE_FALLBACK_INFO_KEY) is not None
        or infos.get(MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_INFO_KEY) is not None
    ):
        raise ValueError("The user-waiver checkpoint lineage drifted")


def create_user_waiver_gate(
    *,
    checkpoint: Path,
    locomotion_report: Path,
    tracking_report: Path,
    onnx_report: Path,
) -> dict[str, Any]:
    """Build the provisional gate; every unwaived check must pass."""

    from mjlab_microban.scripts.evaluate_teleop_v12_checkpoint import _load_actor

    checkpoint = checkpoint.resolve()
    locomotion_report = locomotion_report.resolve()
    tracking_report = tracking_report.resolve()
    onnx_report = onnx_report.resolve()
    checkpoint_sha, iteration, completed, infos = _checkpoint_identity(checkpoint)
    _require_waivable_checkpoint(checkpoint, iteration, completed, checkpoint_sha, infos)
    identity = {
        "sha256": checkpoint_sha,
        "iteration": iteration,
        "completed_updates": completed,
    }
    locomotion = _load_json(locomotion_report)
    tracking = _load_json(tracking_report)
    onnx = _load_json(onnx_report)
    _validate_locomotion_report(locomotion, identity)
    validate_user_waiver_tracking_report(tracking, identity)
    onnx_path, onnx_sha = validate_user_waiver_onnx_report(onnx, identity)
    actor, actor_iteration, _ = _load_actor(checkpoint, device="cpu")
    if actor_iteration != iteration or sha256_file(checkpoint) != checkpoint_sha:
        raise RuntimeError("User-waiver checkpoint changed while it was gated")
    recorded = recorded_corpus_parity(actor, onnx_path, tracking)
    result: dict[str, Any] = {
        "schema_version": 2,
        "gate": "microban_teleop_v12_stage",
        "status": USER_WAIVER_GATE_STATUS,
        "checkpoint": portable_bootstrap_artifact_path(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "iteration": iteration,
        "completed_updates": completed,
        "canonical_boundary": True,
        "checkpoint_kind": "canonical_boundary",
        TELEOP_V12_HOME_POSE_INFO_KEY: deepcopy(infos[TELEOP_V12_HOME_POSE_INFO_KEY]),
        "tracking_profile": USER_WAIVER_TRACKING_PROFILE,
        "adapter_sanitization": infos.get("adapter_sanitization"),
        "reports": {
            "locomotion": portable_bootstrap_artifact_path(locomotion_report),
            "tracking": portable_bootstrap_artifact_path(tracking_report),
            "onnx": portable_bootstrap_artifact_path(onnx_report),
        },
        "report_sha256": {
            "locomotion": sha256_file(locomotion_report),
            "tracking": sha256_file(tracking_report),
            "onnx": sha256_file(onnx_report),
        },
        "onnx": {
            "path": portable_bootstrap_artifact_path(onnx_path),
            "sha256": onnx_sha,
        },
        "tracking_profile_completion_allowance": tracking_profile_completion_allowance(
            FINAL_COMPLETION_ALLOWANCE_PROFILE
        ),
        "user_waiver": user_waiver_record(),
        "user_waiver_sha256": canonical_json_sha256(user_waiver_record()),
        "user_waiver_recorded_corpus_parity": recorded,
    }
    corner_rescue = infos.get(MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY)
    if corner_rescue is not None:
        result[MICROBAN_TELEOP_V12_CORNER_RESCUE_INFO_KEY] = deepcopy(corner_rescue)
    recipe_switch = infos.get(MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_SWITCH_INFO_KEY)
    if recipe_switch is not None:
        result[MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_SWITCH_INFO_KEY] = (
            validate_hand_pose_release_recipe_switch_marker(recipe_switch)
        )
    for marker in (
        pose_release_final_rescue_gate_marker(infos),
        lateral_fidelity_gate_marker(infos),
    ):
        if marker is not None:
            result[marker[0]] = marker[1]
    return result


def validate_user_waiver_gate(gate_path: Path, checkpoint: Path) -> dict[str, Any]:
    """Rebuild the provisional gate from its files; it must be byte-canonical."""

    gate_path = gate_path.resolve()
    checkpoint = checkpoint.resolve()
    gate = _load_json(gate_path)
    if not is_user_waiver_gate(gate) or gate.get("user_waiver") != user_waiver_record():
        raise ValueError("Not the recorded provisional user-waiver gate")
    reports = gate.get("reports")
    report_hashes = gate.get("report_sha256")
    if not isinstance(reports, dict) or not isinstance(report_hashes, dict):
        raise TypeError("User-waiver gate report references are malformed")
    for name in ("locomotion", "tracking", "onnx"):
        report = resolve_bootstrap_artifact_path(reports.get(name, ""))
        if not report.is_file() or sha256_file(report) != report_hashes.get(name):
            raise ValueError(f"User-waiver gate {name} report changed")
    rebuilt = create_user_waiver_gate(
        checkpoint=checkpoint,
        locomotion_report=resolve_bootstrap_artifact_path(reports["locomotion"]),
        tracking_report=resolve_bootstrap_artifact_path(reports["tracking"]),
        onnx_report=resolve_bootstrap_artifact_path(reports["onnx"]),
    )
    if rebuilt != gate:
        raise ValueError("User-waiver gate content is not canonical")
    onnx_path = resolve_bootstrap_artifact_path(gate["onnx"]["path"])
    if not onnx_path.is_file() or sha256_file(onnx_path) != gate["onnx"]["sha256"]:
        raise ValueError("User-waiver gate ONNX artifact changed")
    return gate


def record_onnx_evidence(
    *, checkpoint: Path, expected_sha256: str, onnx_path: Path
) -> dict[str, Any]:
    """teleop_v12_onnx_gate.run_gate, recording (not raising on) a parity miss.

    Same checkpoint load, neutral legacy parity (still raises), export and
    64-sample full-83 corpus as the gate; the report adds the per-sample
    evidence and has ``status == "fail"`` when a bound ratio exceeds 1.
    """

    import copy

    import onnx
    import onnxruntime as ort
    import torch
    from onnx.reference import ReferenceEvaluator
    from tensordict import TensorDict

    from mjlab_microban.scripts.evaluate_teleop_v12_checkpoint import _load_actor
    from mjlab_microban.scripts.teleop_v12_bootstrap_gate import (
        PRISTINE_PARITY_TOLERANCE,
        _export_onnx_atomic,
        _legacy_model,
    )
    from mjlab_microban.tasks.microban_teleop_v12_actor import (
        LEGACY_TO_TELEOP_OBSERVATION_INDEX,
        TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
    )
    from mjlab_microban.tasks.microban_teleop_v12_bootstrap import (
        load_bootstrap_source_state,
        validate_bootstrap_provenance,
    )
    from mjlab_microban.tasks.microban_teleop_v12_runner import (
        TELEOP_V12_BOOTSTRAP_INFO_KEY,
    )

    checkpoint = checkpoint.expanduser().resolve()
    if sha256_file(checkpoint) != expected_sha256:
        raise ValueError("Checkpoint SHA-256 mismatch")
    target, iteration, infos = _load_actor(checkpoint, device="cpu")
    source = _legacy_model()
    source.load_state_dict(
        load_bootstrap_source_state(
            validate_bootstrap_provenance(
                infos.get(TELEOP_V12_BOOTSTRAP_INFO_KEY), verify_files=True
            )
        ),
        strict=True,
    )
    source.eval()
    generator = torch.Generator().manual_seed(20260925)
    neutral = torch.randn(10_000, 83, generator=generator)
    neutral[:, TELEOP_V12_EXTRA_OBSERVATION_COLUMNS] = 0.0
    legacy = neutral[:, [target_index for _, target_index in LEGACY_TO_TELEOP_OBSERVATION_INDEX]]
    with torch.inference_mode():
        expected_actions = copy.deepcopy(source).double()(
            TensorDict({"actor": legacy.double()}, batch_size=[10_000])
        )
        actual_actions = copy.deepcopy(target).double()(
            TensorDict({"actor": neutral.double()}, batch_size=[10_000])
        )
    neutral_max = float(torch.max(torch.abs(actual_actions - expected_actions)).item())
    if neutral_max > PRISTINE_PARITY_TOLERANCE:
        raise ValueError(f"Neutral legacy parity failed: {neutral_max}")
    _export_onnx_atomic(target, onnx_path)
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model, full_check=True)
    reference = ReferenceEvaluator(model)
    runtime = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if runtime.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("ONNX Runtime did not select CPUExecutionProvider only")
    export_model = target.as_onnx(verbose=False).cpu().eval()
    observations = torch.randn(64, 83, generator=generator)
    per: dict[str, list[float]] = {
        "reference_evaluator_absolute_errors": [],
        "onnxruntime_cpu_absolute_errors": [],
        "reference_evaluator_bound_ratios": [],
        "onnxruntime_cpu_bound_ratios": [],
        "maximum_absolute_expected_outputs": [],
    }
    with torch.inference_mode():
        for observation in observations:
            batch = observation.unsqueeze(0)
            expected = export_model(batch).numpy()
            (reference_actual,) = reference.run(None, {"obs": batch.numpy()})
            (runtime_actual,) = runtime.run(None, {"obs": batch.numpy()})
            for prefix, actual in (
                ("reference_evaluator", reference_actual),
                ("onnxruntime_cpu", runtime_actual),
            ):
                per[f"{prefix}_absolute_errors"].append(
                    float(np.max(np.abs(actual - expected)))
                )
                per[f"{prefix}_bound_ratios"].append(
                    parity_bound_ratio(actual, expected, atol=ONNX_PARITY_TOLERANCE)
                )
            per["maximum_absolute_expected_outputs"].append(
                float(np.max(np.abs(expected)))
            )
    maxima = {name: max(values) for name, values in per.items()}
    if not all(math.isfinite(value) for value in maxima.values()):
        raise ValueError("ONNX parity evidence is non-finite")
    failed = [
        name
        for name, ratio in (
            (ONNX_REFERENCE_PARITY_CHECK, maxima["reference_evaluator_bound_ratios"]),
            (ONNX_RUNTIME_CPU_PARITY_CHECK, maxima["onnxruntime_cpu_bound_ratios"]),
        )
        if ratio > 1.0
    ]
    return {
        "schema_version": 1,
        "gate": "microban_teleop_v12_checkpoint_onnx",
        "status": "fail" if failed else "pass",
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": expected_sha256,
            "iteration": iteration,
            "completed_updates": iteration + 1,
        },
        "neutral_legacy_parity": {
            "samples": 10_000,
            "maximum_absolute_error": neutral_max,
            "tolerance": PRISTINE_PARITY_TOLERANCE,
            "teleop_only_columns": "exact_zero",
        },
        "onnx": {
            "path": str(onnx_path.resolve()),
            "sha256": sha256_file(onnx_path),
            "opset": 18,
            "input_shape": [1, 83],
            "output_shape": [1, 18],
            "reference_samples": 64,
            "input_coverage": "deterministic_nonzero_all_83_columns",
            "teleop_only_columns_nonzero": True,
            "reference_evaluator_maximum_absolute_error": maxima[
                "reference_evaluator_absolute_errors"
            ],
            "onnxruntime_cpu_maximum_absolute_error": maxima[
                "onnxruntime_cpu_absolute_errors"
            ],
            "onnxruntime_version": ort.__version__,
            "onnxruntime_providers": runtime.get_providers(),
            "tolerance": ONNX_PARITY_TOLERANCE,
            "relative_tolerance": ONNX_PARITY_RELATIVE_TOLERANCE,
            "parity_rule": ONNX_PARITY_RULE,
            "maximum_absolute_expected_output": maxima[
                "maximum_absolute_expected_outputs"
            ],
            "reference_evaluator_maximum_bound_ratio": maxima[
                "reference_evaluator_bound_ratios"
            ],
            "onnxruntime_cpu_maximum_bound_ratio": maxima[
                "onnxruntime_cpu_bound_ratios"
            ],
            "per_sample": per,
        },
        "parity_failure_recording": True,
        "failed_checks": failed,
    }


# --- packager hooks (export_teleop_v12_deployment, tagged TEMPORARY) --------


def require_user_waiver_final_gate(
    gate: Mapping[str, Any],
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    expected_tracking_profile: str | None,
) -> None:
    """The packager's final-gate identity check for the waiver gate."""

    record = user_waiver_record()
    expected = {
        "schema_version": 2,
        "gate": "microban_teleop_v12_stage",
        "status": USER_WAIVER_GATE_STATUS,
        "checkpoint_sha256": USER_WAIVER_CHECKPOINT_SHA256,
        "iteration": USER_WAIVER_CHECKPOINT_ITERATION,
        "completed_updates": USER_WAIVER_COMPLETED_UPDATES,
        "canonical_boundary": True,
        "checkpoint_kind": "canonical_boundary",
        "tracking_profile": USER_WAIVER_TRACKING_PROFILE,
        "user_waiver": record,
        "user_waiver_sha256": canonical_json_sha256(record),
    }
    mismatches = [name for name, value in expected.items() if gate.get(name) != value]
    if checkpoint_sha256 != USER_WAIVER_CHECKPOINT_SHA256:
        mismatches.append("checkpoint")
    if expected_tracking_profile not in (None, FINAL_COMPLETION_ALLOWANCE_PROFILE):
        mismatches.append("expected_tracking_profile")
    if checkpoint.name != USER_WAIVER_CHECKPOINT_FILENAME:
        mismatches.append("checkpoint_filename")
    if mismatches:
        raise ValueError(
            "Not the recorded provisional user-waiver gate; mismatched fields: "
            + ", ".join(mismatches)
        )


def user_waiver_parity_rule_metadata(onnx_evidence: Mapping[str, Any]) -> dict[str, str]:
    """``_onnx_parity_rule_metadata`` for the waiver: its two values exactly."""

    expected = {
        "parity_rule": ONNX_PARITY_RULE,
        "relative_tolerance": ONNX_PARITY_RELATIVE_TOLERANCE,
        "maximum_absolute_expected_output": USER_WAIVER_MAXIMUM_EXPECTED_OUTPUT,
        "onnxruntime_cpu_maximum_bound_ratio": USER_WAIVER_ONNX_BOUND_RATIO,
    }
    reference_ratio = onnx_evidence.get("reference_evaluator_maximum_bound_ratio")
    if (
        any(onnx_evidence.get(name) != value for name, value in expected.items())
        or isinstance(reference_ratio, bool)
        or not isinstance(reference_ratio, (int, float))
        or not 0.0 <= float(reference_ratio) <= 1.0
    ):
        raise ValueError("User-waiver ONNX parity evidence drifted")
    return {
        "v12_onnx_parity_rule": ONNX_PARITY_RULE,
        "v12_onnx_parity_relative_tolerance": str(ONNX_PARITY_RELATIVE_TOLERANCE),
        "v12_onnx_parity_max_abs_expected_output": str(
            USER_WAIVER_MAXIMUM_EXPECTED_OUTPUT
        ),
        "v12_onnx_reference_max_bound_ratio": str(float(reference_ratio)),
        "v12_onnxruntime_cpu_max_bound_ratio": str(USER_WAIVER_ONNX_BOUND_RATIO),
    }


def user_waiver_final_parity(
    actor: Any, path: Path, *, tolerance: float
) -> dict[str, float]:
    """``_validate_final_parity`` exempting only the waived runtime sample.

    The exempt sample may not exceed its recorded bound ratio; the reference
    evaluator and every other sample keep the normal bound.
    """

    import onnx
    import onnxruntime as ort
    import torch
    from onnx.reference import ReferenceEvaluator

    generator = torch.Generator().manual_seed(20260925)
    torch.randn(10_000, 83, generator=generator)
    observations = torch.randn(64, 83, generator=generator)
    reference = ReferenceEvaluator(onnx.load(path))
    runtime = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    if runtime.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("Deployment parity did not use CPUExecutionProvider only")
    export_model = actor.as_onnx(verbose=False).cpu().eval()
    reference_max = runtime_max = bound_ratio = waived_ratio = 0.0
    with torch.inference_mode():
        for index, observation in enumerate(observations):
            batch = observation.unsqueeze(0)
            expected = export_model(batch).detach().cpu().numpy()
            (reference_actual,) = reference.run(None, {"obs": batch.numpy()})
            (runtime_actual,) = runtime.run(None, {"obs": batch.numpy()})
            reference_max = max(
                reference_max, float(np.max(np.abs(reference_actual - expected)))
            )
            runtime_max = max(
                runtime_max, float(np.max(np.abs(runtime_actual - expected)))
            )
            runtime_ratio = parity_bound_ratio(runtime_actual, expected, atol=tolerance)
            if index == USER_WAIVER_ONNX_SAMPLE_INDEX:
                waived_ratio, runtime_ratio = runtime_ratio, 0.0
            bound_ratio = max(
                bound_ratio,
                parity_bound_ratio(reference_actual, expected, atol=tolerance),
                runtime_ratio,
            )
    values = (reference_max, runtime_max, bound_ratio, waived_ratio)
    if (
        not all(math.isfinite(value) for value in values)
        or bound_ratio > 1.0
        or waived_ratio > USER_WAIVER_ONNX_BOUND_RATIO
    ):
        raise ValueError(f"User-waiver final ONNX parity failed: {values}")
    return {
        "reference_maximum_absolute_error": reference_max,
        "onnxruntime_cpu_maximum_absolute_error": runtime_max,
    }


def user_waiver_package_metadata(
    gate: Mapping[str, Any],
    recorded_parity: Mapping[str, Any] | None,
    *,
    smoke_corpus_sha256: str,
) -> dict[str, str]:
    """The waiver's package metadata (exact record, recorded-corpus parity)."""

    from mjlab_microban.scripts.export_teleop_v12_deployment import _json

    record = user_waiver_record()
    if (
        gate.get("user_waiver") != record
        or recorded_parity is None
        or dict(recorded_parity) != gate.get("user_waiver_recorded_corpus_parity")
        or recorded_parity.get("status") != "pass"
        or recorded_parity.get("corpus_sha256") != smoke_corpus_sha256
    ):
        raise ValueError(
            "Provisional user-waiver package needs the gate's exact record and a "
            "matching passing recorded-corpus parity"
        )
    return {
        "v12_provisional_install": "true",
        "v12_user_waiver_revision": USER_WAIVER_REVISION,
        "v12_user_waiver_json": _json(record),
        "v12_user_waiver_sha256": canonical_json_sha256(record),
        "v12_user_waiver_recorded_corpus_parity_json": _json(dict(recorded_parity)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    record = subparsers.add_parser("record-onnx")
    record.add_argument("checkpoint", type=Path)
    record.add_argument("--onnx", type=Path, required=True)
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--force", action="store_true")
    create = subparsers.add_parser("create")
    create.add_argument("checkpoint", type=Path)
    create.add_argument("locomotion_report", type=Path)
    create.add_argument("tracking_report", type=Path)
    create.add_argument("onnx_report", type=Path)
    create.add_argument("output", type=Path)
    create.add_argument("--force", action="store_true")
    validate = subparsers.add_parser("validate")
    validate.add_argument("gate", type=Path)
    validate.add_argument("checkpoint", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "record-onnx":
        if not args.force and (args.onnx.exists() or args.output.exists()):
            raise FileExistsError("Output exists (pass --force)")
        report = record_onnx_evidence(
            checkpoint=args.checkpoint,
            expected_sha256=USER_WAIVER_CHECKPOINT_SHA256,
            onnx_path=args.onnx.expanduser().resolve(),
        )
        publish_json_atomic(args.output, report)
        print(json.dumps(report, sort_keys=True), flush=True)
        return 0
    if args.command == "create":
        if args.output.exists() and not args.force:
            raise FileExistsError("Gate exists (pass --force)")
        gate = create_user_waiver_gate(
            checkpoint=args.checkpoint,
            locomotion_report=args.locomotion_report,
            tracking_report=args.tracking_report,
            onnx_report=args.onnx_report,
        )
        publish_json_atomic(args.output, gate)
    else:
        gate = validate_user_waiver_gate(args.gate, args.checkpoint)
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
