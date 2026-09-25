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
