"""Dump every HOME-dependent value of a mjlab_microban tree, for side-by-side checks.

Used by tests/test_home_pose_any_trunk.py and to record the reference dumps in
tests/fixtures/home_equivalence/ from the branches each HOME was trained on:

    # in a scratch worktree of the reference branch, with this tree's venv:
    PYTHONHASHSEED=0 PYTHONPATH=<ref>/src python tests/home_equivalence.py dump OUT.json <ref>
    python tests/home_equivalence.py digest OUT.json REF.json   # sha256 per key

``dump`` imports every module of mjlab_microban.robot / .tasks / .scripts and
records each UPPER_CASE module constant, the zero-argument HOME-derived
functions (hand FK metadata, HOME stamps, v12 marker,
parity corpus) and the repr of every registered task's env / play env / RL
config and runner class.  Values are normalised: memory addresses and the tree
path are removed and sets are sorted.  It needs no GPU (CUDA_VISIBLE_DEVICES='').
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import inspect
import io
import json
import pkgutil
import re
import sys
import warnings
from pathlib import Path

ADDRESS = re.compile(r" at 0x[0-9a-f]+")

# New dataclass fields that a vertical-trunk HOME leaves at their (no-op)
# defaults: the centered branch's configs never had them.  Removed from the
# current repr before it is compared with the centered reference.
CENTERED_DEFAULT_FIELDS = (
    ("trunk_pitch=0.0, ", ""),
    (", trunk_pitch=0.0", ""),
    (", lf_rb_probability=0.9", ""),
)


def _value_repr(value: object, root: str) -> str:
    try:
        import numpy as np
        import torch

        if isinstance(value, torch.Tensor):
            return "tensor" + repr(value.tolist())
        if isinstance(value, np.ndarray):
            return "ndarray" + repr(value.tolist())
    except Exception:  # noqa: BLE001
        pass
    if isinstance(value, (set, frozenset)):
        return f"{type(value).__name__}" + repr(sorted(repr(item) for item in value))
    return ADDRESS.sub("", repr(value)).replace(root, "<ROOT>")


def dump(root: Path) -> dict[str, str | None]:
    warnings.filterwarnings("ignore")
    root = root.resolve()
    with contextlib.redirect_stdout(io.StringIO()):
        import mjlab_microban
        import mjlab_microban.tasks  # noqa: F401 - registers the tasks
    if not Path(mjlab_microban.__file__).resolve().is_relative_to(root):
        raise RuntimeError(f"mjlab_microban was imported from {mjlab_microban.__file__}, not {root}")
    text_root = str(root)
    out: dict[str, str | None] = {}
    names = []
    for prefix in ("mjlab_microban.robot", "mjlab_microban.tasks", "mjlab_microban.scripts"):
        package = importlib.import_module(prefix)
        names += [info.name for info in pkgutil.iter_modules(package.__path__, prefix + ".")]
    for name in sorted(names):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                module = importlib.import_module(name)
        except BaseException as error:  # noqa: BLE001
            out[f"import:{name}"] = f"ERROR {type(error).__name__}"
            continue
        for key, value in sorted(vars(module).items()):
            if key.startswith("__") or not (key.isupper() or (key.startswith("_") and key[1:].isupper())):
                continue
            if inspect.ismodule(value) or inspect.isclass(value) or inspect.isfunction(value):
                continue
            out[f"const:{name}.{key}"] = _value_repr(value, text_root)
    calls = (
        ("mjlab_microban.robot.microban_hand_fk", "microban_hand_fk_metadata"),
        ("mjlab_microban.robot.microban_hand_fk", "microban_reachable_hand_evaluation_offsets"),
        ("mjlab_microban.tasks.microban_getup_runner", "getup_home_pose"),
        # The walking export's HOME stamp (export_walk_onnx.walk_home_pose on the
        # reference branches; every policy's stamp since microban-policy-1).
        ("mjlab_microban.policy_contract", "home_pose_stamp"),
        ("mjlab_microban.tasks.microban_teleop_v12_home_pose", "teleop_v12_home_pose_marker"),
    )
    for module_name, function_name in calls:
        key = f"call:{module_name}.{function_name}()"
        try:
            value = getattr(importlib.import_module(module_name), function_name)()
            out[key] = _value_repr(value, text_root)
        except BaseException as error:  # noqa: BLE001
            out[key] = f"ERROR {type(error).__name__}"
    try:
        import numpy as np
        from mjlab_microban.tasks.microban_policy_export import deterministic_teleop_parity_inputs

        rows = np.ascontiguousarray(deterministic_teleop_parity_inputs())
        out["call:deterministic_teleop_parity_inputs()"] = hashlib.sha256(rows.tobytes()).hexdigest()
    except BaseException as error:  # noqa: BLE001
        out["call:deterministic_teleop_parity_inputs()"] = f"ERROR {type(error).__name__}"
    from mjlab.tasks import registry

    for task in registry.list_tasks():
        entry = registry._REGISTRY[task]  # noqa: SLF001
        out[f"task:{task}:env"] = _value_repr(entry.env_cfg, text_root)
        out[f"task:{task}:play"] = _value_repr(entry.play_env_cfg, text_root)
        out[f"task:{task}:rl"] = _value_repr(entry.rl_cfg, text_root)
        runner = entry.runner_cls
        out[f"task:{task}:runner"] = None if runner is None else f"{runner.__module__}.{runner.__qualname__}"
    return out


def digest(values: dict[str, str | None]) -> dict[str, str]:
    return {
        key: hashlib.sha256(json.dumps(value).encode("utf-8")).hexdigest()
        for key, value in sorted(values.items())
    }


def without_centered_default_fields(value: str | None) -> str | None:
    if value is None:
        return None
    for old, new in CENTERED_DEFAULT_FIELDS:
        value = value.replace(old, new)
    return value


def main(argv: list[str]) -> int:
    command = argv[0]
    if command == "dump":
        Path(argv[1]).write_text(json.dumps(dump(Path(argv[2])), indent=1, sort_keys=True))
        return 0
    if command == "digest":
        values = json.loads(Path(argv[1]).read_text())
        Path(argv[2]).write_text(json.dumps(digest(values), indent=1, sort_keys=True) + "\n")
        return 0
    raise SystemExit(f"unknown command {command!r}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
