"""Fail-closed tests for exact model10099 canonical-task live simulation."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from mjlab_microban.scripts.live_pico_teleop_sim import (
    V12_CANONICAL_TASK,
    _construct_checkpoint_consumer_runner,
    _runtime_task,
    _selected_checkpoint,
    _validate_deadline_canary_live_authority,
    build_parser,
)
from mjlab_microban.tasks.microban_teleop_v12_runner import (
    MicrobanTeleopV12DeadlineCanarySimulationConsumer,
    make_teleop_v12_deadline_canary_simulation_consumer,
)

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/run_pico_v12_deadline_canary_sim.sh"


class DeadlineCanaryLiveCliTest(unittest.TestCase):
    def test_dedicated_option_selects_only_canonical_v12_task(self) -> None:
        checkpoint = Path("model_10099.pt")
        receipt = Path("post_canary_receipt.json")
        args = build_parser().parse_args(
            [
                "--v12-deadline-canary-checkpoint",
                str(checkpoint),
                "--v12-deadline-canary-acceptance-receipt",
                str(receipt),
                "--v12-deadline-canary-acceptance-receipt-sha256",
                "a" * 64,
                "--input",
                "pico-app",
            ]
        )
        self.assertEqual(_selected_checkpoint(args), checkpoint)
        self.assertEqual(
            _runtime_task(
                args.checkpoint,
                args.v12_preview_checkpoint,
                args.unaccepted_sim_preview_checkpoint,
                args.v12_controller_preview_checkpoint,
                args.v12_deadline_canary_checkpoint,
            ),
            V12_CANONICAL_TASK,
        )
        self.assertEqual(V12_CANONICAL_TASK, "Mjlab-Teleop-V12-Microban")

        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                [
                    "--checkpoint",
                    "generic.pt",
                    "--v12-deadline-canary-checkpoint",
                    str(checkpoint),
                ]
            )
        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                [
                    "--v12-preview-checkpoint",
                    "preview.pt",
                    "--v12-deadline-canary-checkpoint",
                    str(checkpoint),
                ]
            )

    def test_dedicated_consumer_factory_requires_canonical_task(self) -> None:
        module = "mjlab_microban.scripts.live_pico_teleop_sim"
        with patch(
            f"{module}.make_teleop_v12_deadline_canary_simulation_consumer",
            return_value="consumer",
        ) as factory:
            result = _construct_checkpoint_consumer_runner(
                "env",
                "cfg",
                "cuda:0",
                runtime_task=V12_CANONICAL_TASK,
                deadline_canary_mode=True,
            )
        self.assertEqual(result, "consumer")
        factory.assert_called_once_with("env", "cfg", "cuda:0")

        with self.assertRaisesRegex(ValueError, "canonical contract-v12 task"):
            _construct_checkpoint_consumer_runner(
                "env",
                "cfg",
                "cuda:0",
                runtime_task="Mjlab-Teleop-Microban",
                deadline_canary_mode=True,
            )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            _construct_checkpoint_consumer_runner(
                "env",
                "cfg",
                "cuda:0",
                runtime_task=V12_CANONICAL_TASK,
                deadline_canary_mode=True,
                unaccepted_preview_mode=True,
            )


class DeadlineCanaryLiveAuthorityTest(unittest.TestCase):
    def _artifacts(self, root: Path) -> tuple[Path, Path, str, str]:
        checkpoint = root / "model_10099.pt"
        torch.save(
            {
                "iter": 10_099,
                "infos": {"env_state": {"common_step_counter": 242_400}},
            },
            checkpoint,
        )
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        receipt = root / "post_canary_receipt.json"
        receipt.write_text(
            json.dumps(
                {
                    "gate": "microban_teleop_v12_deadline_post_canary_acceptance",
                    "status": "pass",
                    "checkpoint": {"sha256": checkpoint_sha},
                    "strict_failure_report": {"path": "/evidence/strict.json"},
                    "fallback_tracking_report": {"path": "/evidence/fallback.json"},
                    "full_stage_gate": {"path": "/evidence/gate.json"},
                }
            )
        )
        receipt_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
        return checkpoint, receipt, checkpoint_sha, receipt_sha

    def test_authority_hashes_then_validates_and_keeps_checkpoint_bytes(self) -> None:
        module = "mjlab_microban.scripts.live_pico_teleop_sim"
        with TemporaryDirectory() as directory:
            checkpoint, receipt, checkpoint_sha, receipt_sha = self._artifacts(
                Path(directory)
            )
            with (
                patch(
                    f"{module}.MICROBAN_TELEOP_V12_DEADLINE_CANARY_CHECKPOINT_SHA256",
                    checkpoint_sha,
                ),
                patch(
                    f"{module}.MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_RECEIPT_SHA256",
                    receipt_sha,
                ),
                patch(
                    f"{module}.validate_deadline_post_canary_receipt_payload",
                    return_value={"status": "pass"},
                ) as receipt_validator,
                patch(
                    f"{module}.validate_post_canary_receipt",
                    return_value=json.loads(receipt.read_bytes()),
                ) as full_receipt_validator,
                patch(
                    f"{module}.validate_deadline_fallback_canary_payload",
                    return_value={},
                ) as checkpoint_validator,
            ):
                authority = _validate_deadline_canary_live_authority(
                    checkpoint=checkpoint,
                    acceptance_receipt=receipt,
                    expected_receipt_sha256=receipt_sha,
                )

            self.assertEqual(authority.checkpoint_sha256, checkpoint_sha)
            self.assertEqual(authority.checkpoint_bytes, checkpoint.read_bytes())
            self.assertEqual(authority.acceptance_receipt_sha256, receipt_sha)
            receipt_validator.assert_called_once()
            self.assertEqual(
                receipt_validator.call_args.kwargs["checkpoint_sha256"],
                checkpoint_sha,
            )
            self.assertEqual(
                receipt_validator.call_args.kwargs["receipt_sha256"],
                receipt_sha,
            )
            full_receipt_validator.assert_called_once_with(
                receipt=receipt,
                checkpoint=checkpoint,
                strict_tracking_report=Path("/evidence/strict.json"),
                fallback_tracking_report=Path("/evidence/fallback.json"),
                stage_gate=Path("/evidence/gate.json"),
            )
            checkpoint_validator.assert_called_once()
            self.assertEqual(
                checkpoint_validator.call_args.kwargs,
                {
                    "verify_parent_files": True,
                    "checkpoint_sha256": checkpoint_sha,
                },
            )

    def test_authority_rejects_missing_changed_or_nonexact_artifacts(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires its post-canary PASS"):
            _validate_deadline_canary_live_authority(
                checkpoint=Path("model_10099.pt"),
                acceptance_receipt=None,
                expected_receipt_sha256=None,
            )
        with TemporaryDirectory() as directory:
            checkpoint, receipt, checkpoint_sha, receipt_sha = self._artifacts(
                Path(directory)
            )
            with self.assertRaisesRegex(ValueError, "not the pinned PASS receipt"):
                _validate_deadline_canary_live_authority(
                    checkpoint=checkpoint,
                    acceptance_receipt=receipt,
                    expected_receipt_sha256="0" * 64,
                )
            module = "mjlab_microban.scripts.live_pico_teleop_sim"
            with patch(
                f"{module}.MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_RECEIPT_SHA256",
                "0" * 64,
            ), self.assertRaisesRegex(ValueError, "receipt SHA-256 mismatch"):
                _validate_deadline_canary_live_authority(
                    checkpoint=checkpoint,
                    acceptance_receipt=receipt,
                    expected_receipt_sha256="0" * 64,
                )
            with patch(
                f"{module}.MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_RECEIPT_SHA256",
                receipt_sha,
            ), self.assertRaisesRegex(ValueError, "pinned model10099"):
                _validate_deadline_canary_live_authority(
                    checkpoint=checkpoint,
                    acceptance_receipt=receipt,
                    expected_receipt_sha256=receipt_sha,
                )

            symlink = Path(directory) / "receipt-link.json"
            symlink.symlink_to(receipt)
            with (
                patch(
                    f"{module}.MICROBAN_TELEOP_V12_DEADLINE_CANARY_CHECKPOINT_SHA256",
                    checkpoint_sha,
                ),
                patch(
                    f"{module}.MICROBAN_TELEOP_V12_DEADLINE_POST_CANARY_RECEIPT_SHA256",
                    receipt_sha,
                ),
                self.assertRaisesRegex(ValueError, "cannot be a symlink"),
            ):
                _validate_deadline_canary_live_authority(
                    checkpoint=checkpoint,
                    acceptance_receipt=symlink,
                    expected_receipt_sha256=receipt_sha,
                )


class DeadlineCanaryReadOnlyRunnerTest(unittest.TestCase):
    def test_consumer_class_is_immutable_and_has_no_mutating_api(self) -> None:
        cfg = {
            "checkpoint_consumer_mode": True,
            "simulation_preview_mode": False,
            "resume": False,
        }
        base = (
            "mjlab_microban.tasks.microban_teleop_v12_runner."
            "MicrobanTeleopV12OnPolicyRunner.__init__"
        )
        with patch(base, return_value=None):
            consumer = MicrobanTeleopV12DeadlineCanarySimulationConsumer(
                "env", cfg, device="cuda:0"
            )
        self.assertTrue(consumer.require_immutable_checkpoint_bytes)
        self.assertTrue(consumer.allow_deadline_canary_consumer)
        for method in (
            consumer.learn,
            consumer.save,
            consumer.export_policy_to_onnx,
        ):
            with self.subTest(method=method.__name__), self.assertRaises(RuntimeError):
                method()

        with patch(base, return_value=None), self.assertRaisesRegex(
            ValueError, "immutable, read-only"
        ):
            MicrobanTeleopV12DeadlineCanarySimulationConsumer(
                "env", {**cfg, "resume": True}, device="cuda:0"
            )

    def test_factory_forces_actor_consumer_flags(self) -> None:
        agent_cfg = {"resume": True, "simulation_preview_mode": True}
        target = (
            "mjlab_microban.tasks.microban_teleop_v12_runner."
            "MicrobanTeleopV12DeadlineCanarySimulationConsumer"
        )
        with patch(target, return_value="consumer") as consumer_class:
            result = make_teleop_v12_deadline_canary_simulation_consumer(
                "env", agent_cfg, "cuda:0"
            )
        self.assertEqual(result, "consumer")
        cfg = consumer_class.call_args.args[1]
        self.assertIs(cfg["checkpoint_consumer_mode"], True)
        self.assertIs(cfg["simulation_preview_mode"], False)
        self.assertIs(cfg["resume"], False)
        self.assertIsNone(consumer_class.call_args.kwargs.get("log_dir"))

    def test_wrapper_is_exact_and_has_no_generic_or_physical_route(self) -> None:
        text = WRAPPER.read_text()
        self.assertIn("EXPECTED_CHECKPOINT_SHA256", text)
        self.assertIn("EXPECTED_RECEIPT_SHA256", text)
        self.assertIn(
            "f7fcf511ba75f326cd94a2fc21a05360fa9f7925849fc2bce91709fd8678f27b",
            text,
        )
        self.assertIn("--v12-deadline-canary-checkpoint", text)
        self.assertNotIn(" --checkpoint ", text)
        self.assertNotIn("--robot", text)
        self.assertNotIn("--send", text)
        self.assertNotIn("train_microban", text)
        self.assertNotIn("export_policy", text)


if __name__ == "__main__":
    unittest.main()
