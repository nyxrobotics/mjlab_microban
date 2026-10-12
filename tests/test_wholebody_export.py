"""The whole-body ONNX: its input, ONNX Runtime == torch, the contract metadata and the env's frame layout."""

from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from tensordict import TensorDict

from mjlab_microban.policy_contract import ACTION_JOINT_NAMES, PolicyContractError, joint_ranges, sha256_file
from mjlab_microban.robot.home_pose import HOME
from mjlab_microban.wholebody import contract
from mjlab_microban.wholebody.env_cfg import ACTOR_GROUP, WholebodyRlCfg, actor_terms, make_wholebody_env_cfg
from mjlab_microban.wholebody.model import TemporalCNNModel

WIDTH = contract.OBSERVATION_WIDTH
HISTORY = contract.HISTORY_LENGTH


def _untrained_actor() -> TemporalCNNModel:
    torch.manual_seed(0)
    cfg = WholebodyRlCfg.actor
    sample = TensorDict({ACTOR_GROUP: torch.randn(64, HISTORY, WIDTH)}, batch_size=[64])
    model = TemporalCNNModel(
        sample,
        {"actor": [ACTOR_GROUP]},
        "actor",
        len(ACTION_JOINT_NAMES),
        hidden_dims=cfg.hidden_dims,
        activation=cfg.activation,
        obs_normalization=cfg.obs_normalization,
        distribution_cfg=deepcopy(cfg.distribution_cfg),
        cnn_cfg=deepcopy(cfg.cnn_cfg),
    )
    # Non-trivial normalizers, so the export must carry them.
    model.update_normalization(TensorDict(
        {ACTOR_GROUP: torch.cat([-1.0 + 0.5 * torch.randn(512, HISTORY - 1, WIDTH),
                                 3.0 + 2.0 * torch.randn(512, 1, WIDTH)], dim=1)},
        batch_size=[512],
    ))
    return model.eval()


def _possible_frame(rng: np.random.Generator) -> np.ndarray:
    frame = 0.05 * rng.standard_normal(WIDTH)
    terms = {name: slice(sum(w for _, w in contract.OBSERVATION_SCHEMA[:i]),
                         sum(w for _, w in contract.OBSERVATION_SCHEMA[: i + 1]))
             for i, (name, _) in enumerate(contract.OBSERVATION_SCHEMA)}
    frame[terms["projected_gravity"]] = HOME.projected_gravity
    frame[terms["ref_projected_gravity_b"]] = HOME.projected_gravity
    frame[terms["ref_anchor_height"]] = HOME.root_pos[2]
    return frame


def test_export_matches_torch_and_carries_the_contract(tmp_path):
    model = _untrained_actor()
    path = tmp_path / "wholebody.onnx"
    contract.export_policy_to_onnx(model, path)
    contract.check_onnx_io(path)
    assert [i.name for i in onnx.load(str(path)).graph.input] == list(contract.INPUT_NAMES)

    rng = np.random.default_rng(1)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for _ in range(4):
        history = rng.standard_normal((1, HISTORY, WIDTH)).astype(np.float32)
        expected = model(TensorDict({ACTOR_GROUP: torch.from_numpy(history)}, batch_size=[1])).detach().numpy()
        actual = session.run(None, {"obs_history": history})[0]
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)

    rows = [np.stack([_possible_frame(rng) for _ in range(HISTORY)]).tolist() for _ in range(8)]
    batch = TensorDict({ACTOR_GROUP: torch.tensor(rows, dtype=torch.float32)}, batch_size=[8])
    actions = model(batch).double().tolist()
    metadata = contract.contract_metadata(
        joint_names=list(HOME.joint_pos_rad),
        checkpoint=tmp_path / "model_4999.pt",
        checkpoint_sha256="a" * 64,
        benchmark_report_sha256="b" * 64,
        eval_only_benchmark_report_sha256="c" * 64,
        self_test_rows=rows,
        self_test_actions=actions,
        dry_run=True,
    )
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx

    attach_metadata_to_onnx(str(path), metadata)
    assert contract.check_self_test(path, metadata) <= 1.0
    stored = {entry.key: entry.value for entry in onnx.load(str(path)).metadata_props}
    assert stored["microban_policy_contract"] == "microban-wholebody-1"
    assert stored["microban_policy_kind"] == "wholebody"
    assert stored["microban_recipe"] == "microban-wholebody-teleopit-track-1"
    assert stored["input_names"] == "obs_history"
    assert stored["history_length"] == "10"
    assert stored["neck_control"] == "reference_clamped_slewed"
    assert stored["reference_time"] == "one_control_period_ahead"
    assert stored["action_joint_names"].split(",") == list(ACTION_JOINT_NAMES)
    assert sum(width for _, width in json.loads(stored["observation_schema_json"])) == WIDTH == 118
    assert stored["eval_only_benchmark_report_sha256"] == "c" * 64
    assert stored["dry_run_not_deployable"] == "true"


def test_export_asks_for_both_sets_started_at_home(tmp_path):
    # The fall line holds only on an evaluation-only set (the PICO recordings),
    # and only a start at HOME is the robot's.
    from mjlab_microban.scripts.wholebody_export import benchmark_sha256

    def report(name: str, eval_only_set: bool = False, start: str | None = "home", passed: bool = True,
               sha: str = "a" * 64):
        metadata = {"checkpoint_sha256": sha, "eval_only_set": eval_only_set, "start": start}
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({
            "metadata": {key: value for key, value in metadata.items() if value is not None},
            "verdicts": {"final": {"pass": passed}},
        }))
        return path

    train, pico = report("train"), report("pico", eval_only_set=True)
    assert benchmark_sha256(train, "a" * 64, False, dry_run=False) == sha256_file(train)
    assert benchmark_sha256(pico, "a" * 64, True, dry_run=False) == sha256_file(pico)
    for path, eval_only_set, problem in (
        (train, True, "not the benchmark on an evaluation-only set"),  # the training set's report has no fall line
        (pico, False, "not the benchmark on the training kinds' set"),
        (report("reference", start="reference"), False, "does not start the pieces at HOME"),
        (report("earlier", start=None), False, "does not start the pieces at HOME"),  # before the start was recorded
        (report("failed", eval_only_set=True, passed=False), True, "does not pass"),
        (report("other", sha="b" * 64), False, "not the benchmark of this checkpoint"),
    ):
        with pytest.raises(PolicyContractError, match=problem):
            benchmark_sha256(path, "a" * 64, eval_only_set, dry_run=False)
    # A dry run takes any report of the checkpoint.
    assert benchmark_sha256(report("failed2", start="reference", passed=False), "a" * 64, True, dry_run=True)
    with pytest.raises(PolicyContractError, match="this checkpoint"):
        benchmark_sha256(report("other2", sha="b" * 64), "a" * 64, False, dry_run=True)


def test_env_frame_layout_is_the_contract_schema():
    # The actor's terms in order, its joint terms over the robot's joints in
    # MJCF order, its reference terms over the policy's joints.
    assert [name for name, _ in contract.OBSERVATION_SCHEMA] == list(actor_terms())
    assert list(contract.OBSERVATION_JOINT_NAMES) == list(joint_ranges())
    cfg = make_wholebody_env_cfg()
    assert cfg.commands["motion"].joint_names == contract.REFERENCE_JOINT_NAMES
    assert cfg.observations[ACTOR_GROUP].history_length == contract.HISTORY_LENGTH
    assert WholebodyRlCfg.obs_groups["actor"] == (ACTOR_GROUP,)
