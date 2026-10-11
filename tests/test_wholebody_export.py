"""The whole-body ONNX: inputs, ONNX Runtime == torch, and the contract metadata."""

from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import onnx
import onnxruntime as ort
import torch
from tensordict import TensorDict

from mjlab_microban.policy_contract import ACTION_JOINT_NAMES
from mjlab_microban.robot.home_pose import HOME
from mjlab_microban.wholebody import contract
from mjlab_microban.wholebody.env_cfg import WholebodyRlCfg
from mjlab_microban.wholebody.model import TemporalCNNModel

WIDTH = contract.OBSERVATION_WIDTH
HISTORY = contract.HISTORY_LENGTH


def _untrained_actor() -> TemporalCNNModel:
    torch.manual_seed(0)
    cfg = WholebodyRlCfg.actor
    sample = TensorDict({"actor": torch.randn(64, WIDTH), "actor_history": torch.randn(64, HISTORY, WIDTH)},
                        batch_size=[64])
    model = TemporalCNNModel(
        sample,
        {"actor": ["actor", "actor_history"]},
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
        {"actor": 3.0 + 2.0 * torch.randn(512, WIDTH), "actor_history": -1.0 + 0.5 * torch.randn(512, HISTORY, WIDTH)},
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
        obs = rng.standard_normal((1, WIDTH)).astype(np.float32)
        history = rng.standard_normal((1, HISTORY, WIDTH)).astype(np.float32)
        expected = model(TensorDict({"actor": torch.from_numpy(obs), "actor_history": torch.from_numpy(history)},
                                    batch_size=[1])).detach().numpy()
        actual = session.run(None, {"obs": obs, "obs_history": history})[0]
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)

    rows = []
    for _ in range(8):
        history = np.stack([_possible_frame(rng) for _ in range(HISTORY)])
        rows.append((history[-1].tolist(), history.tolist()))
    batch = TensorDict({"actor": torch.tensor([r[0] for r in rows], dtype=torch.float32),
                        "actor_history": torch.tensor([r[1] for r in rows], dtype=torch.float32)}, batch_size=[8])
    actions = model(batch).double().tolist()
    metadata = contract.contract_metadata(
        joint_names=list(HOME.joint_pos_rad),
        checkpoint=tmp_path / "model_4999.pt",
        checkpoint_sha256="a" * 64,
        benchmark_report_sha256="b" * 64,
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
    assert stored["input_names"] == "obs,obs_history"
    assert stored["history_length"] == "10"
    assert stored["neck_control"] == "hmd"
    assert stored["action_joint_names"].split(",") == list(ACTION_JOINT_NAMES)
    assert sum(width for _, width in json.loads(stored["observation_schema_json"])) == WIDTH == 118
    assert stored["dry_run_not_deployable"] == "true"
