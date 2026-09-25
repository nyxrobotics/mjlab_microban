"""Contract-v9 bounded safe-velocity to teleop bootstrap tests."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from mjlab_microban.robot.microban_constants import MICROBAN_ROBOT_CFG
from mjlab_microban.scripts import (
    evaluate_teleop_checkpoint,
    export_teleop_onnx,
    live_pico_teleop_sim,
)
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION,
    MicrobanTeleopOnPolicyRunner,
    validate_safe_velocity_actor_bootstrap_info,
)
from mjlab_microban.tasks.microban_safe_velocity_checkpoint import (
    inspect_safe_velocity_checkpoint,
)
from mjlab_microban.tasks.microban_safe_velocity_env_cfg import (
    microban_safe_velocity_action_delta_bounds,
    microban_safe_velocity_initial_action_std,
)
from mjlab_microban.tasks.microban_safe_velocity_mdp import (
    MICROBAN_SAFE_VELOCITY_RECIPE_INFO_KEY,
    MICROBAN_SAFE_VELOCITY_RECIPE_REVISION,
    MicrobanSafeVelocityBoundedGaussianDistribution,
)
from mjlab_microban.tasks.microban_teleop_bootstrap import (
    SAFE_VELOCITY_COPIED_DISTRIBUTION_KEYS,
    SAFE_VELOCITY_NEW_TELEOP_OBSERVATION_COLUMNS,
    TELEOP_ACTOR_OBSERVATION_WIDTH,
    VELOCITY_ACTOR_OBSERVATION_WIDTH,
    bootstrap_teleop_actor_state,
    expand_velocity_observation_to_teleop,
    load_safe_velocity_actor_bootstrap,
    serialize_safe_velocity_actor_bootstrap_provenance,
    validate_safe_velocity_acceptance_receipt,
)
from mjlab_microban.tasks.microban_teleop_env_cfg import (
    MicrobanTeleopRlCfg,
    make_microban_teleop_env_cfg,
    microban_teleop_action_delta_bounds,
)


def _actor(observation_width: int) -> MLPModel:
    lower, upper = microban_safe_velocity_action_delta_bounds()
    observation = TensorDict(
        {"actor": torch.zeros((1, observation_width), dtype=torch.float32)},
        batch_size=[1],
    )
    return MLPModel(
        obs=observation,
        obs_groups={"actor": ["actor"]},
        obs_set="actor",
        output_dim=18,
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=False,
        distribution_cfg={
            "class_name": MicrobanSafeVelocityBoundedGaussianDistribution,
            "init_std": microban_safe_velocity_initial_action_std(),
            "lower_bound": lower,
            "upper_bound": upper,
            "std_type": "log",
        },
    )


def _mlp_output(
    state: dict[str, torch.Tensor], observation: torch.Tensor
) -> torch.Tensor:
    value = observation
    for layer in (0, 2, 4):
        value = torch.nn.functional.elu(
            torch.nn.functional.linear(
                value, state[f"mlp.{layer}.weight"], state[f"mlp.{layer}.bias"]
            )
        )
    return torch.nn.functional.linear(value, state["mlp.6.weight"], state["mlp.6.bias"])


class SafeVelocityBootstrapMappingTest(unittest.TestCase):
    def test_mapping_preserves_source_output_and_zeroes_all_new_columns(self) -> None:
        torch.manual_seed(20260925)
        source = _actor(VELOCITY_ACTOR_OBSERVATION_WIDTH).state_dict()
        target = _actor(TELEOP_ACTOR_OBSERVATION_WIDTH).state_dict()
        mapped = bootstrap_teleop_actor_state(target, source)

        velocity_observation = torch.randn((7, VELOCITY_ACTOR_OBSERVATION_WIDTH))
        teleop_observation = expand_velocity_observation_to_teleop(velocity_observation)
        torch.testing.assert_close(
            _mlp_output(source, velocity_observation),
            _mlp_output(mapped, teleop_observation),
            rtol=0.0,
            atol=0.0,
        )
        new_columns = list(SAFE_VELOCITY_NEW_TELEOP_OBSERVATION_COLUMNS)
        self.assertEqual(len(new_columns), 20)
        self.assertTrue(
            torch.equal(
                mapped["mlp.0.weight"][:, new_columns],
                torch.zeros_like(mapped["mlp.0.weight"][:, new_columns]),
            )
        )
        self.assertFalse(any(key.startswith("obs_normalizer.") for key in mapped))

    def test_complete_mlp_head_and_distribution_are_copied(self) -> None:
        source = _actor(VELOCITY_ACTOR_OBSERVATION_WIDTH).state_dict()
        target = _actor(TELEOP_ACTOR_OBSERVATION_WIDTH).state_dict()
        mapped = bootstrap_teleop_actor_state(target, source)

        for key, value in source.items():
            if key == "mlp.0.weight":
                continue
            torch.testing.assert_close(mapped[key], value, rtol=0.0, atol=0.0)
        self.assertEqual(
            {key for key in mapped if key.startswith("distribution.")},
            set(SAFE_VELOCITY_COPIED_DISTRIBUTION_KEYS),
        )

    def test_normalizer_or_distribution_drift_fails_closed(self) -> None:
        source = _actor(VELOCITY_ACTOR_OBSERVATION_WIDTH).state_dict()
        target = _actor(TELEOP_ACTOR_OBSERVATION_WIDTH).state_dict()

        normalized = dict(source)
        normalized["obs_normalizer._mean"] = torch.zeros((1, 63))
        with self.assertRaisesRegex(ValueError, "raw observations"):
            bootstrap_teleop_actor_state(target, normalized)

        changed_target = {key: value.clone() for key, value in target.items()}
        changed_target["distribution.mean_upper_bound"][0] += 1.0e-6
        with self.assertRaisesRegex(ValueError, "contracts differ"):
            bootstrap_teleop_actor_state(changed_target, source)


class SafeVelocityBootstrapLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _safe_checkpoint(self, iteration: int = 7) -> tuple[Path, str, Path]:
        path = self.root / f"model_{iteration}.pt"
        torch.save(
            {
                "actor_state_dict": _actor(63).state_dict(),
                "critic_state_dict": {"must_not_copy": torch.tensor(91.0)},
                "optimizer_state_dict": {"must_not_copy": 92},
                "iter": iteration,
                "infos": {
                    MICROBAN_SAFE_VELOCITY_RECIPE_INFO_KEY: (
                        MICROBAN_SAFE_VELOCITY_RECIPE_REVISION
                    ),
                    "env_state": {"common_step_counter": (iteration + 1) * 24},
                },
            },
            path,
        )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        identity = inspect_safe_velocity_checkpoint(path, expected_sha256=digest)
        receipt = self.root / f"safe_velocity_acceptance_{iteration}.json"
        receipt.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "gate": "microban_safe_velocity_fixed_forward_v3",
                    "checkpoint": {
                        "path": str(identity.path),
                        "sha256": identity.sha256,
                        "iteration": identity.iteration,
                        "checkpoint_schema_version": identity.schema_version,
                        "recipe_revision": identity.recipe_revision,
                        "actor_topology": list(identity.actor_topology),
                        "actor_obs_normalization": identity.actor_obs_normalization,
                        "observation_schema": [
                            list(item) for item in identity.observation_schema
                        ],
                        "action_joint_names": list(identity.action_joint_names),
                    },
                    "configuration": {
                        "device": "cuda:0",
                        "num_envs": 64,
                        "steps_requested": 200,
                        "steps_executed": 200,
                        "seed": 42,
                        "command_vx_m_s": 0.08,
                        "actual_command_exactly_verified_each_step": True,
                        "step_dt_s": 0.02,
                        "guard_lookahead_s": 0.12,
                        "guard_margin_ratio": 0.05,
                        "absolute_clip_max_tensor_error_rad": 0.0,
                    },
                    "thresholds": {
                        "completion_fraction_min": 0.95,
                        "fall_fraction_max": 0.0,
                        "nonfinite_fraction_max": 0.0,
                        "forward_velocity_p05_m_s_min": 0.01,
                        "forward_displacement_p05_m_min": 0.02,
                        "actual_soft_limit_violation_rad_max": 1.0e-6,
                        "actual_lookahead_soft_limit_violation_rad_max": 1.0e-6,
                        "target_clip_rad_max": 1.0e-7,
                    },
                    "metrics": {
                        "completion_fraction": 1.0,
                        "fall_fraction": 0.0,
                        "nonfinite_fraction": 0.0,
                        "forward_velocity_p05_m_s": 0.03,
                        "forward_velocity_median_m_s": 0.03,
                        "forward_displacement_p05_m": 0.03,
                        "forward_displacement_median_m": 0.03,
                        "maximum_actual_soft_limit_violation_rad": 0.0,
                        "maximum_actual_lookahead_soft_limit_violation_rad": 0.0,
                        "maximum_preferred_margin_lookahead_excess_rad": 0.0,
                        "maximum_target_clip_rad": 0.0,
                        "minimum_root_height_m": 0.2,
                    },
                    "checks": {
                        name: True
                        for name in (
                            "completion",
                            "no_falls",
                            "finite",
                            "forward_velocity",
                            "forward_displacement",
                            "actual_soft_limits",
                            "actual_lookahead_soft_limits",
                            "absolute_target_clip",
                        )
                    },
                    "status": "pass",
                    "summary": {"passed": True, "failed_checks": []},
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return path, digest, receipt

    def test_strict_loader_and_serialized_provenance(self) -> None:
        checkpoint, digest, receipt = self._safe_checkpoint()
        target = _actor(83)
        before_target = deepcopy(target.state_dict())
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            load_safe_velocity_actor_bootstrap(target, checkpoint, "0" * 64, receipt)
        for key, value in target.state_dict().items():
            torch.testing.assert_close(value, before_target[key])

        provenance = load_safe_velocity_actor_bootstrap(
            target, checkpoint, digest, receipt
        )
        info = serialize_safe_velocity_actor_bootstrap_provenance(provenance)
        validate_safe_velocity_actor_bootstrap_info(info, verify_source_checkpoint=True)
        self.assertEqual(info["source_checkpoint_sha256"], digest)
        self.assertEqual(info["source_checkpoint_iteration"], 7)
        self.assertEqual(info["source_acceptance_receipt_path"], str(receipt.resolve()))
        self.assertEqual(info["critic_copied"], False)
        self.assertEqual(info["optimizer_copied"], False)
        self.assertEqual(info["iteration_copied"], False)

        tampered = deepcopy(info)
        tampered["observation_index_mapping"][0] = [0, 1]  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "contract v9"):
            validate_safe_velocity_actor_bootstrap_info(
                tampered, verify_source_checkpoint=False
            )

        receipt.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validate_safe_velocity_actor_bootstrap_info(info)

    def test_legacy_normalized_unbounded_actor_is_rejected(self) -> None:
        checkpoint = self.root / "model_999.pt"
        legacy_state = {
            "mlp.0.weight": torch.zeros((512, 63)),
            "obs_normalizer._mean": torch.zeros((1, 63)),
            "distribution.std_param": torch.ones((18,)),
        }
        torch.save(
            {"actor_state_dict": legacy_state, "iter": 999},
            checkpoint,
        )
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "recipe mismatch"):
            load_safe_velocity_actor_bootstrap(
                _actor(83), checkpoint, digest, self.root / "missing-receipt.json"
            )

    def test_acceptance_receipt_is_semantically_fail_closed(self) -> None:
        checkpoint, digest, receipt = self._safe_checkpoint()
        identity = inspect_safe_velocity_checkpoint(checkpoint, expected_sha256=digest)
        valid = json.loads(receipt.read_text(encoding="utf-8"))
        invalid_reports = []
        for mutation in (
            lambda report: report.__setitem__("status", "fail"),
            lambda report: report["summary"].__setitem__("passed", False),
            lambda report: report["checks"].__setitem__("no_falls", False),
            lambda report: report["configuration"].__setitem__(
                "actual_command_exactly_verified_each_step", False
            ),
            lambda report: report["checkpoint"].__setitem__("sha256", "0" * 64),
        ):
            candidate = deepcopy(valid)
            mutation(candidate)
            invalid_reports.append(candidate)
        for index, report in enumerate(invalid_reports):
            with self.subTest(index=index):
                receipt.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaises((TypeError, ValueError)):
                    validate_safe_velocity_acceptance_receipt(receipt, identity)

    def test_real_failed_receipt_schema_and_status_only_forgery_are_rejected(
        self,
    ) -> None:
        checkpoint, digest, synthetic_receipt = self._safe_checkpoint()
        identity = inspect_safe_velocity_checkpoint(checkpoint, expected_sha256=digest)
        fixture = (
            Path(__file__).parent
            / "fixtures"
            / "microban_safe_velocity_v4_model_500_failed.json"
        )
        report = json.loads(fixture.read_text(encoding="utf-8"))
        self.assertIn("actual_lookahead_soft_limits", report["checks"])
        self.assertNotIn("preferred_lookahead_guard", report["checks"])
        # Rebind this historical v2 failure to the current v3 wire contract
        # while preserving a genuine performance failure for the forgery test.
        report["gate"] = "microban_safe_velocity_fixed_forward_v3"
        report["thresholds"]["forward_velocity_p05_m_s_min"] = 0.01
        report["metrics"]["forward_velocity_p05_m_s"] = 0.009

        synthetic = json.loads(synthetic_receipt.read_text(encoding="utf-8"))
        report["checkpoint"] = synthetic["checkpoint"]
        rebound = self.root / "real_failed_receipt_rebound.json"
        rebound.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "did not pass all checks"):
            validate_safe_velocity_acceptance_receipt(rebound, identity)

        report["status"] = "pass"
        rebound.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "status is inconsistent"):
            validate_safe_velocity_acceptance_receipt(rebound, identity)


class TeleopV9ConfigurationTest(unittest.TestCase):
    def test_actor_is_raw_critic_normalized_and_bounds_match_source(self) -> None:
        self.assertEqual(MICROBAN_TELEOP_TRAINING_CONTRACT_VERSION, "9")
        self.assertFalse(MicrobanTeleopRlCfg.actor.obs_normalization)
        self.assertTrue(MicrobanTeleopRlCfg.critic.obs_normalization)
        self.assertIs(
            MicrobanTeleopRlCfg.actor.distribution_cfg["class_name"],
            MicrobanSafeVelocityBoundedGaussianDistribution,
        )
        self.assertEqual(
            microban_teleop_action_delta_bounds(),
            microban_safe_velocity_action_delta_bounds(),
        )

    def test_teleop_home_uses_zero_shoulder_pitch(self) -> None:
        cfg = make_microban_teleop_env_cfg()
        self.assertGreaterEqual(cfg.sim.nconmax, 512)
        self.assertGreaterEqual(cfg.sim.njmax, 2048)
        pose = cfg.scene.entities["robot"].init_state.joint_pos
        assert pose is not None
        self.assertEqual(pose["left_shoulder_pitch"], 0.0)
        self.assertEqual(pose["right_shoulder_pitch"], 0.0)
        robot_home = MICROBAN_ROBOT_CFG.init_state.joint_pos
        assert robot_home is not None
        for name in (
            "left_shoulder_pitch",
            "right_shoulder_pitch",
            "left_shoulder_roll",
            "right_shoulder_roll",
            "left_elbow",
            "right_elbow",
        ):
            self.assertEqual(pose[name], robot_home[name])

    def test_runner_rejects_missing_source_legacy_options_and_source_on_resume(
        self,
    ) -> None:
        env = SimpleNamespace(clip_actions=None, num_actions=18)
        with self.assertRaisesRegex(ValueError, "fresh contract-v9"):
            MicrobanTeleopOnPolicyRunner(env, {})
        with self.assertRaisesRegex(ValueError, "rejects legacy"):
            MicrobanTeleopOnPolicyRunner(
                env,
                {
                    "bootstrap_velocity_checkpoint": "/tmp/model_999.pt",
                    "bootstrap_velocity_checkpoint_sha256": "0" * 64,
                },
            )
        with self.assertRaisesRegex(ValueError, "fresh-run only"):
            MicrobanTeleopOnPolicyRunner(
                env,
                {
                    "resume": True,
                    "safe_velocity_checkpoint": "/tmp/model_7.pt",
                    "safe_velocity_checkpoint_sha256": "0" * 64,
                    "safe_velocity_acceptance_receipt": "/tmp/receipt.json",
                },
            )

    def test_all_checkpoint_consumers_select_actor_load_only_mode(self) -> None:
        factories = (
            export_teleop_onnx._construct_checkpoint_consumer_runner,
            evaluate_teleop_checkpoint._construct_checkpoint_consumer_runner,
            live_pico_teleop_sim._construct_checkpoint_consumer_runner,
        )
        for factory in factories:

            class _FakeRunner:
                def __init__(
                    self, env: object, config: dict[str, object], device: str
                ) -> None:
                    self.env = env
                    self.config = config
                    self.device = device

            module = __import__(factory.__module__, fromlist=["dummy"])
            agent_cfg = deepcopy(MicrobanTeleopRlCfg)
            with patch.object(module, "load_runner_cls", return_value=_FakeRunner):
                runner = factory("env", agent_cfg, "cpu")
            self.assertIsInstance(runner, _FakeRunner)
            config = runner.config
            self.assertIs(config["checkpoint_consumer_mode"], True)
            self.assertFalse(config["resume"])
            self.assertIsNone(config["safe_velocity_checkpoint"])
            self.assertEqual(runner.env, "env")
            self.assertEqual(runner.device, "cpu")

    def test_checkpoint_consumer_mode_cannot_resume_train_save_or_full_load(
        self,
    ) -> None:
        env = SimpleNamespace(clip_actions=None, num_actions=18)
        with self.assertRaisesRegex(ValueError, "cannot resume training"):
            MicrobanTeleopOnPolicyRunner(
                env, {"checkpoint_consumer_mode": True, "resume": True}
            )
        with self.assertRaisesRegex(ValueError, "cannot accept fresh"):
            MicrobanTeleopOnPolicyRunner(
                env,
                {
                    "checkpoint_consumer_mode": True,
                    "safe_velocity_checkpoint": "/tmp/model_7.pt",
                    "safe_velocity_checkpoint_sha256": "0" * 64,
                    "safe_velocity_acceptance_receipt": "/tmp/receipt.json",
                },
            )

        runner = object.__new__(MicrobanTeleopOnPolicyRunner)
        runner.checkpoint_consumer_mode = True
        runner.safe_velocity_bootstrap_provenance = None
        with self.assertRaisesRegex(ValueError, "explicit actor-only"):
            runner.load("/missing/model_7.pt")
        with self.assertRaisesRegex(RuntimeError, "cannot train"):
            runner.learn(1)
        with self.assertRaisesRegex(RuntimeError, "cannot save"):
            runner.save("/missing/model_7.pt")

    def test_wrapper_requires_explicit_source_and_has_valid_shell_syntax(self) -> None:
        root = Path(__file__).resolve().parents[1]
        wrapper = root / "scripts" / "train_microban_teleop_v9.sh"
        subprocess.run(["bash", "-n", str(wrapper)], check=True)
        help_result = subprocess.run(
            [str(wrapper), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("SAFE_MODEL_PT SAFE_SHA256 PASS_RECEIPT_JSON", help_result.stdout)
        missing = subprocess.run(
            [str(wrapper), "start"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(missing.returncode, 2)
        oversized = subprocess.run(
            [str(wrapper), "start", "/missing/model_0.pt", "0" * 64, "/missing.json"],
            check=False,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "MICROBAN_TELEOP_NUM_ENVS": "4096",
            },
        )
        self.assertEqual(oversized.returncode, 2)
        self.assertIn("requires MICROBAN_TELEOP_NUM_ENVS=2048", oversized.stderr)
        script = wrapper.read_text(encoding="utf-8")
        self.assertIn("target_boundary=1500", script)
        self.assertIn("18000 20000", script)
        self.assertIn("MICROBAN_TELEOP_PROVENANCE_MODE=canonical_v9_stage", script)
        self.assertIn("MICROBAN_TELEOP_STAGE_START_BOUNDARY", script)
        self.assertIn("MICROBAN_TELEOP_PARENT_GATE_SHA256", script)
        self.assertIn("MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_PATH", script)
        self.assertIn("MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_SHA256", script)
        self.assertIn("MICROBAN_TELEOP_RESUME_SOURCE_CHECKPOINT_ITERATION", script)
        self.assertIn("evaluate_microban_teleop_v9_stage.sh", script)


if __name__ == "__main__":
    unittest.main()
