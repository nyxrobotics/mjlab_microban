from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_microban_teleop_v12_relocated_gate.py"
SPEC = importlib.util.spec_from_file_location("relocated_gate_validator", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

ArtifactRelocator = MODULE.ArtifactRelocator
RelocationError = MODULE.RelocationError


def _artifact(checkout: Path, relative: str) -> Path:
    path = checkout / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"authenticated artifact")
    return path


def test_relocator_maps_repo_checkout_and_recorded_absolute_paths(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    relative = "artifacts/teleop_v12_gates/stage.json"
    expected = _artifact(checkout, relative)
    recorded = tmp_path / "old-checkout"
    relocator = ArtifactRelocator(checkout, [recorded])

    assert relocator.resolve(f"repo://{relative}") == expected
    assert relocator.resolve(expected) == expected
    assert relocator.resolve(recorded / relative) == expected
    assert relocator.portable(recorded / relative) == f"repo://{relative}"


@pytest.mark.parametrize(
    "value",
    [
        "artifacts/teleop_v12_gates/stage.json",
        "repo:///artifacts/teleop_v12_gates/stage.json",
        "repo://artifacts/teleop_v12_gates/../stage.json",
        "repo://artifacts//teleop_v12_gates/stage.json",
        "repo://src/mjlab_microban/scripts/teleop_v12_stage.py",
        "repo://artifacts/not_teleop/stage.json",
        "repo://logs/rsl_rl/not_microban/stage.json",
    ],
)
def test_relocator_rejects_relative_escape_and_non_data_paths(
    tmp_path: Path, value: str
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    relocator = ArtifactRelocator(checkout)

    with pytest.raises(RelocationError):
        relocator.resolve(value)


def test_relocator_rejects_unapproved_absolute_and_missing_paths(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    relocator = ArtifactRelocator(checkout, [tmp_path / "recorded"])

    with pytest.raises(RelocationError, match="outside checkout and recorded roots"):
        relocator.resolve(tmp_path / "other" / "artifacts/teleop_v12_gates/x.json")
    with pytest.raises(RelocationError, match="does not exist"):
        relocator.resolve("repo://artifacts/teleop_v12_gates/missing.json")


def test_relocator_rejects_symlink_file_and_parent(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    target = _artifact(checkout, "artifacts/teleop_v12_gates/target.json")
    linked_file = checkout / "artifacts/teleop_v12_gates/linked.json"
    linked_file.symlink_to(target.name)
    real_directory = checkout / "artifacts/teleop_v12_real"
    real_directory.mkdir(parents=True)
    (real_directory / "nested.json").write_bytes(b"nested")
    linked_directory = checkout / "artifacts/teleop_v12_linked"
    linked_directory.symlink_to(real_directory.name, target_is_directory=True)
    relocator = ArtifactRelocator(checkout)

    with pytest.raises(RelocationError, match="symlink"):
        relocator.resolve("repo://artifacts/teleop_v12_gates/linked.json")
    with pytest.raises(RelocationError, match="symlink"):
        relocator.resolve("repo://artifacts/teleop_v12_linked/nested.json")


def test_relocator_allows_only_regular_files(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    directory = checkout / "artifacts/teleop_v12_gates/a-directory"
    directory.mkdir(parents=True)
    relocator = ArtifactRelocator(checkout)

    with pytest.raises(RelocationError, match="not a regular file"):
        relocator.resolve("repo://artifacts/teleop_v12_gates/a-directory")


def test_recorded_roots_must_be_absolute_and_non_symlinked(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    actual = tmp_path / "actual"
    actual.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(actual, target_is_directory=True)

    with pytest.raises(RelocationError, match="absolute"):
        ArtifactRelocator(checkout, [Path("relative")])
    with pytest.raises(RelocationError, match="Symlink"):
        ArtifactRelocator(checkout, [linked])


def test_artifact_log_allowlist(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    relative = "logs/rsl_rl/mjlab_microban_teleop_v12/run/model.pt"
    expected = _artifact(checkout, relative)
    relocator = ArtifactRelocator(checkout)

    assert relocator.resolve(f"repo://{relative}") == expected


@pytest.mark.parametrize(
    "relative",
    [
        "checkpoints/xc330_velocity/model_14999.pt",
        "artifacts/legacy_teleop_probe/model_14999_teleop83_raw_9x300.json",
    ],
)
def test_exact_pinned_bootstrap_evidence_allowlist(
    tmp_path: Path, relative: str
) -> None:
    checkout = tmp_path / "checkout"
    expected = _artifact(checkout, relative)
    relocator = ArtifactRelocator(checkout)

    assert relocator.resolve(f"repo://{relative}") == expected


def test_module_origin_must_be_real_file_under_checkout_src(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    source = checkout / "src/mjlab_microban/module.py"
    source.parent.mkdir(parents=True)
    source.write_text("", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("", encoding="utf-8")
    linked = source.with_name("linked.py")
    linked.symlink_to(source.name)

    assert MODULE._module_is_from_checkout(
        SimpleNamespace(__file__=str(source)), checkout
    )
    assert not MODULE._module_is_from_checkout(
        SimpleNamespace(__file__=str(outside)), checkout
    )
    assert not MODULE._module_is_from_checkout(
        SimpleNamespace(__file__=str(linked)), checkout
    )


def test_install_patches_bootstrap_before_stage_and_all_loaded_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "checkout"
    (checkout / "src").mkdir(parents=True)
    relocator = ArtifactRelocator(checkout)
    bootstrap = ModuleType("mjlab_microban.tasks.microban_teleop_v12_bootstrap")
    stage = ModuleType("mjlab_microban.scripts.teleop_v12_stage")
    copied_alias = ModuleType("mjlab_microban.tasks.copied_alias")
    original_resolver = lambda value: Path(value)
    original_portable = lambda value: str(value)
    for module in (bootstrap, stage, copied_alias):
        module.resolve_bootstrap_artifact_path = original_resolver
        module.portable_bootstrap_artifact_path = original_portable

    module_names = {
        bootstrap.__name__: bootstrap,
        stage.__name__: stage,
        copied_alias.__name__: copied_alias,
    }
    for name, module in module_names.items():
        monkeypatch.setitem(sys.modules, name, module)

    saved: list[tuple[ModuleType, str, object]] = []
    for name, module in tuple(sys.modules.items()):
        if module is None or not name.startswith("mjlab_microban"):
            continue
        for attribute in (
            "resolve_bootstrap_artifact_path",
            "portable_bootstrap_artifact_path",
        ):
            if hasattr(module, attribute):
                saved.append((module, attribute, getattr(module, attribute)))

    stage_import_saw_prepatch = False

    def fake_import(name: str) -> ModuleType:
        nonlocal stage_import_saw_prepatch
        if name == bootstrap.__name__:
            return bootstrap
        if name == stage.__name__:
            resolver = bootstrap.resolve_bootstrap_artifact_path
            portable = bootstrap.portable_bootstrap_artifact_path
            stage_import_saw_prepatch = (
                getattr(resolver, "__self__", None) is relocator
                and getattr(portable, "__self__", None) is relocator
            )
            return stage
        raise AssertionError(name)

    monkeypatch.setattr(MODULE.importlib, "import_module", fake_import)
    monkeypatch.setattr(MODULE, "_module_is_from_checkout", lambda *_: True)
    try:
        result = MODULE.install_relocated_resolvers(relocator)
        assert result is stage
        assert stage_import_saw_prepatch
        for patched in (bootstrap, stage, copied_alias):
            assert patched.resolve_bootstrap_artifact_path.__self__ is relocator
            assert patched.portable_bootstrap_artifact_path.__self__ is relocator
    finally:
        for module, attribute, value in saved:
            setattr(module, attribute, value)


def test_main_preserves_repo_uri_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    checkout = tmp_path / "checkout"
    gate_relative = "artifacts/teleop_v12_gates/gate.json"
    checkpoint_relative = "logs/rsl_rl/mjlab_microban_teleop_v12/run/model.pt"
    gate = _artifact(checkout, gate_relative)
    checkpoint = _artifact(checkout, checkpoint_relative)
    called: list[tuple[Path, Path]] = []

    def validate_gate(gate_path: Path, checkpoint_path: Path) -> dict[str, str]:
        called.append((gate_path, checkpoint_path))
        return {"status": "pass"}

    monkeypatch.setattr(
        MODULE,
        "install_relocated_resolvers",
        lambda _: SimpleNamespace(validate_gate=validate_gate),
    )

    assert (
        MODULE.main(
            [
                "--checkout-root",
                str(checkout),
                f"repo://{gate_relative}",
                f"repo://{checkpoint_relative}",
            ]
        )
        == 0
    )
    assert called == [(gate, checkpoint)]
    assert "MICROBAN_TELEOP_V12_RELOCATED_GATE=PASS" in capsys.readouterr().out
