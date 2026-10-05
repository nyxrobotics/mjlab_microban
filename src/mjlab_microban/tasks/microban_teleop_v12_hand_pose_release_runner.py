"""Runner for the active-hand arm pose-release recipe.

A fresh start of this task records the new recipe revision (release-eligible).
Resuming a v11 checkpoint is a recipe switch and needs one of two explicit
options:

* ``release_recipe_switch_gate`` (+ its SHA-256): the release-eligible switch.
  Only the pinned, gated canonical model_7099 is accepted; its stage gate must
  pass the stage-gate validator for exactly that checkpoint.  Every save
  carries the ``release_eligible: True`` marker naming the parent and gate,
  which stage gates, evaluators and the exporter re-validate on every load.
* ``experimental_recipe_switch``: any v11 parent, marked
  ``release_eligible: False``; evidence only, never gated or exported.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from mjlab_microban.tasks.microban_teleop_v12_bootstrap import (
    portable_bootstrap_artifact_path,
    sha256_file,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import (
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION,
    MICROBAN_TELEOP_V12_RECIPE_REVISION,
)
from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release import (
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_SWITCH_INFO_KEY,
    active_hand_arm_released_posture,
)
from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release_lineage import (
    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_SWITCH_INFO_KEY,
    hand_pose_release_recipe_switch_marker,
    validate_hand_pose_release_recipe_switch_marker,
    validate_hand_pose_release_switch_parent_payload,
    verify_hand_pose_release_switch_parent,
)
from mjlab_microban.tasks.microban_teleop_v12_runner import (
    MicrobanTeleopV12OnPolicyRunner,
)

HAND_POSE_RELEASE_SWITCH_SCHEMA_VERSION = 1


def validate_hand_pose_release_switch_marker(value: Any) -> dict[str, Any]:
    """Validate the not-for-release marker of a v11 -> pose-release switch."""

    if not isinstance(value, dict):
        raise TypeError("Hand pose-release switch marker is malformed")
    expected_keys = {
        "schema_version",
        "release_eligible",
        "parent_recipe_revision",
        "recipe_revision",
        "parent_checkpoint_sha256",
        "parent_iteration",
    }
    if set(value) != expected_keys:
        raise ValueError("Hand pose-release switch marker keys drifted")
    sha = value["parent_checkpoint_sha256"]
    if (
        value["schema_version"] != HAND_POSE_RELEASE_SWITCH_SCHEMA_VERSION
        or value["release_eligible"] is not False
        or value["parent_recipe_revision"] != MICROBAN_TELEOP_V12_RECIPE_REVISION
        or value["recipe_revision"]
        != MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
        or not isinstance(sha, str)
        or len(sha) != 64
        or any(c not in "0123456789abcdef" for c in sha)
        or not isinstance(value["parent_iteration"], int)
        or isinstance(value["parent_iteration"], bool)
    ):
        raise ValueError("Hand pose-release switch marker drifted")
    return deepcopy(value)


class MicrobanTeleopV12HandPoseReleaseOnPolicyRunner(
    MicrobanTeleopV12OnPolicyRunner
):
    """V12 runner that trains and records the pose-release recipe."""

    accepts_hand_pose_release_recipe = True

    def __init__(self, env, train_cfg: dict, *args: Any, **kwargs: Any) -> None:
        cfg = deepcopy(train_cfg)
        switch = cfg.pop("experimental_recipe_switch", False)
        release_gate = cfg.pop("release_recipe_switch_gate", "") or ""
        release_gate_sha256 = cfg.pop("release_recipe_switch_gate_sha256", "") or ""
        if type(switch) is not bool:
            raise TypeError("experimental_recipe_switch must be a boolean")
        if not isinstance(release_gate, str) or not isinstance(
            release_gate_sha256, str
        ):
            raise TypeError("release_recipe_switch_gate options must be strings")
        release_switch = bool(release_gate or release_gate_sha256)
        if release_switch and (
            not release_gate
            or len(release_gate_sha256) != 64
            or any(c not in "0123456789abcdef" for c in release_gate_sha256)
        ):
            raise ValueError(
                "release_recipe_switch_gate requires the gate path and its SHA-256"
            )
        if (switch or release_switch) and not cfg.get("resume", False):
            raise ValueError("A recipe switch applies only to a resume")
        if switch and release_switch:
            raise ValueError("Experimental and release recipe switches are exclusive")
        if cfg.get("deadline_fallback_resume") or cfg.get("simulation_preview_mode"):
            raise ValueError("Hand pose release has no deadline or preview route")
        self.experimental_recipe_switch = switch
        self.release_recipe_switch_gate = release_gate if release_switch else None
        self.release_recipe_switch_gate_sha256 = (
            release_gate_sha256 if release_switch else None
        )
        # (infos key, marker) carried by every save of this run, or None.
        self.teleop_v12_hand_pose_release_switch: dict[str, Any] | None = None
        self.teleop_v12_hand_pose_release_switch_key: str | None = None
        super().__init__(env, cfg, *args, **kwargs)
        self._assert_hand_pose_release_environment()

    def _assert_hand_pose_release_environment(self) -> None:
        pose = self.env.unwrapped.reward_manager.get_term_cfg("pose")
        if not isinstance(pose.func, active_hand_arm_released_posture) or (
            pose.params.get("hand_command_name") != "hand_target"
        ):
            raise RuntimeError("Hand pose-release reward term is not installed")

    def load(
        self,
        path: str | bytes,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        if isinstance(path, bytes):
            raise TypeError("Hand pose-release training loads a filesystem checkpoint")
        resolved = Path(path).expanduser().resolve(strict=True)
        before = sha256_file(resolved)
        payload = torch.load(resolved, map_location="cpu", weights_only=False)
        infos = payload.get("infos") if isinstance(payload, dict) else None
        if not isinstance(infos, dict):
            raise TypeError("Hand pose-release parent payload is malformed")
        recipe = infos.get("microban_teleop_recipe_revision")
        existing = infos.get(MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_SWITCH_INFO_KEY)
        existing_release = infos.get(
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_SWITCH_INFO_KEY
        )
        release_switch = self.release_recipe_switch_gate is not None
        key: str | None
        if recipe == MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION:
            if self.experimental_recipe_switch or release_switch:
                raise ValueError(
                    "A recipe switch applies only to a v11 parent; this checkpoint "
                    "already uses the hand pose-release recipe"
                )
            if existing is not None and existing_release is not None:
                raise ValueError("A checkpoint cannot carry both switch markers")
            if existing_release is not None:
                # Parent files and gate are re-validated by the base load.
                key = MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_SWITCH_INFO_KEY
                marker = validate_hand_pose_release_recipe_switch_marker(
                    existing_release
                )
            elif existing is not None:
                key = MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_SWITCH_INFO_KEY
                marker = validate_hand_pose_release_switch_marker(existing)
            else:
                key = None
                marker = None
        elif recipe == MICROBAN_TELEOP_V12_RECIPE_REVISION and release_switch:
            if existing is not None or existing_release is not None:
                raise ValueError("A v11 checkpoint cannot carry a switch marker")
            validate_hand_pose_release_switch_parent_payload(
                payload, checkpoint_sha256=before
            )
            assert self.release_recipe_switch_gate is not None
            gate_path = Path(self.release_recipe_switch_gate).expanduser().resolve(
                strict=True
            )
            if sha256_file(gate_path) != self.release_recipe_switch_gate_sha256:
                raise ValueError("Release recipe switch gate SHA-256 mismatch")
            key = MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_SWITCH_INFO_KEY
            marker = hand_pose_release_recipe_switch_marker(
                parent_checkpoint_path=portable_bootstrap_artifact_path(resolved),
                parent_checkpoint_sha256=before,
                parent_stage_gate_path=portable_bootstrap_artifact_path(gate_path),
                parent_stage_gate_sha256=self.release_recipe_switch_gate_sha256,
            )
            # Same validator every consumer runs: hashes + full stage gate.
            verify_hand_pose_release_switch_parent(marker)
        elif recipe == MICROBAN_TELEOP_V12_RECIPE_REVISION:
            if not self.experimental_recipe_switch:
                raise ValueError(
                    "Resuming a v11 checkpoint into the hand pose-release recipe "
                    "mixes recipes; pass --agent.release-recipe-switch-gate "
                    "(gated canonical model_7099 only) or "
                    "--agent.experimental-recipe-switch True for an experiment, "
                    "or start a fresh chain"
                )
            if existing is not None or existing_release is not None:
                raise ValueError("A v11 checkpoint cannot carry a switch marker")
            key = MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_SWITCH_INFO_KEY
            marker = {
                "schema_version": HAND_POSE_RELEASE_SWITCH_SCHEMA_VERSION,
                "release_eligible": False,
                "parent_recipe_revision": MICROBAN_TELEOP_V12_RECIPE_REVISION,
                "recipe_revision": (
                    MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
                ),
                "parent_checkpoint_sha256": before,
                "parent_iteration": payload.get("iter"),
            }
            validate_hand_pose_release_switch_marker(marker)
        else:
            raise ValueError("Hand pose release resumes only v11 or its own recipe")
        loaded = super().load(
            str(resolved), load_cfg=load_cfg, strict=strict, map_location=map_location
        )
        if sha256_file(resolved) != before:
            raise ValueError("Hand pose-release parent changed while loading")
        if release_switch and recipe == MICROBAN_TELEOP_V12_RECIPE_REVISION:
            assert self.release_recipe_switch_gate is not None
            if sha256_file(Path(self.release_recipe_switch_gate).resolve()) != (
                self.release_recipe_switch_gate_sha256
            ):
                raise ValueError("Release recipe switch gate changed while loading")
        self.teleop_v12_hand_pose_release_switch = marker
        self.teleop_v12_hand_pose_release_switch_key = key
        self._assert_hand_pose_release_environment()
        return loaded

    def _contract_infos(self, infos: dict | None = None) -> dict:
        self._assert_hand_pose_release_environment()
        result = super()._contract_infos(infos)
        result["microban_teleop_recipe_revision"] = (
            MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_REVISION
        )
        marker = self.teleop_v12_hand_pose_release_switch
        key = self.teleop_v12_hand_pose_release_switch_key
        if marker is not None:
            if key == MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_RECIPE_SWITCH_INFO_KEY:
                marker = validate_hand_pose_release_recipe_switch_marker(marker)
            elif key == MICROBAN_TELEOP_V12_HAND_POSE_RELEASE_SWITCH_INFO_KEY:
                marker = validate_hand_pose_release_switch_marker(marker)
            else:
                raise RuntimeError("Hand pose-release switch marker key is unbound")
            result[key] = deepcopy(marker)
        return result
