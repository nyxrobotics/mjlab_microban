from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "artifacts"
    / "teleop_v12_releases"
    / "microban_teleop_v12_full_chain_manifest.json"
)


def test_release_chain_manifest_has_all_boundaries_and_canaries() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["schema"] == "microban_teleop_v12_release_chain_v1"
    assert payload["status"] == "pass"
    assert payload["recorded_project_roots"] == [
        "/home/kanade/Git-projects/mjlab_microban_v8j",
        "/home/kanade/Git-projects/mjlab_microban_v12_deferred",
    ]
    assert [stage["completed_updates"] for stage in payload["stages"]] == [
        3000,
        3100,
        7000,
        7100,
        10000,
        10100,
        15000,
    ]
    assert [stage["canonical_boundary"] for stage in payload["stages"]] == [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
    ]
    assert all(len(stage["validator_commit"]) == 40 for stage in payload["stages"])
    assert payload["deployment"]["input_shape"] == [1, 83]
    assert payload["deployment"]["output_shape"] == [1, 18]
    assert payload["deployment"]["packaged_policy"].startswith("repo://")
    migration = payload["bilateral_migration"]
    assert migration["status"] == "pass"
    assert migration["raw_checkpoint_sha256"] == (
        "16c9b9d19df6513851b3da26228ae612512fdb2d894542741f0474e4762691c7"
    )
    assert migration["migrated_checkpoint_sha256"] == (
        "83d915b8646ae5c70aab4d367ec9f0516e9b42fe4b6ec5a0977482b81018459b"
    )


def test_release_chain_archived_paths_hashes_and_receipt() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify_microban_teleop_v12_release_chain.py"),
            "--paths-only",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "MICROBAN_TELEOP_V12_RELEASE_CHAIN=PASS" in result.stdout
    for updates in (3000, 3100, 7000, 7100, 10000, 10100, 15000):
        assert f"archived update={updates}" in result.stdout


def test_release_chain_full_pinned_validators_and_packaged_runtime() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify_microban_teleop_v12_release_chain.py"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "packaged ONNX CPU runtime samples=16" in result.stdout
    assert "relocated bilateral migration and corner parent" in result.stdout
    assert result.stdout.count("[PASS] pinned validator") == 7
    assert "MICROBAN_TELEOP_V12_RELEASE_CHAIN=PASS" in result.stdout
