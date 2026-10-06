"""Configuration, state, input hashes and the job runner of the pipeline.

Only the standard library and PyYAML: the command runs under the system
Python as well as under ``uv run``; every training and evaluation is a
``uv run --locked`` subprocess.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

REPO = Path(__file__).resolve().parents[3]
HOME_YAML = REPO / "config" / "home_pose.yaml"
PIPELINE_YAML = REPO / "config" / "pipeline.yaml"
LOG_ROOT = REPO / "logs" / "rsl_rl"
UV = ["uv", "run", "--locked"]
UV_ONNX = [*UV, "--with", "onnxruntime", "--with", "protobuf<7"]

EXIT_DONE, EXIT_FAILED, EXIT_STALL, EXIT_INPUT, EXIT_BUSY = 0, 1, 2, 3, 4
EXIT_INTERRUPTED = 130


class PipelineError(Exception):
    """A step stopped the run; ``code`` is the command's exit code."""

    def __init__(self, message: str, code: int = EXIT_FAILED) -> None:
        super().__init__(message)
        self.code = code


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_inputs(paths: Iterable[str], extra: object = None) -> str:
    """SHA-256 over the files below ``paths`` (repo-relative globs) and ``extra``.

    A directory includes every file below it; ``__pycache__`` is skipped.
    """

    digest = hashlib.sha256()
    files: set[Path] = set()
    for pattern in paths:
        for match in sorted(REPO.glob(pattern)):
            if match.is_dir():
                files.update(p for p in match.rglob("*") if p.is_file())
            elif match.is_file():
                files.add(match)
    for path in sorted(files):
        if "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(REPO)).encode())
        digest.update(b"\0")
        digest.update(sha256(path).encode())
    digest.update(json.dumps(extra, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def _merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(dry: bool, path: Path = PIPELINE_YAML) -> dict[str, Any]:
    """config/pipeline.yaml, with its ``dry:`` section merged in for a dry run."""

    raw = yaml.safe_load(path.read_text())
    dry_section = raw.pop("dry", {}) or {}
    return _merge(raw, dry_section) if dry else raw


def model_index(name: str) -> int | None:
    match = re.fullmatch(r"model_(\d+)\.pt", name)
    return int(match.group(1)) if match else None


def runs_of(experiment: str, label: str) -> list[Path]:
    """The run directories of ``label`` (mjlab names them <time>_<label>), oldest first."""

    root = LOG_ROOT / experiment
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.endswith("_" + label))


def checkpoints(run: Path) -> list[int]:
    if not run.is_dir():
        return []
    return sorted(i for i in (model_index(p.name) for p in run.iterdir()) if i is not None)


def latest_checkpoint(experiment: str, label: str) -> Path | None:
    """The newest checkpoint of ``label`` over all its run directories (resumes included)."""

    best: tuple[int, float, Path] | None = None
    for run in runs_of(experiment, label):
        for index in checkpoints(run):
            path = run / f"model_{index}.pt"
            key = (index, path.stat().st_mtime, path)
            if best is None or key[:2] > best[:2]:
                best = key
    return None if best is None else best[2]


def find_checkpoint(experiment: str, label: str, index: int) -> Path | None:
    for run in reversed(runs_of(experiment, label)):
        path = run / f"model_{index}.pt"
        if path.is_file():
            return path
    return None


class State:
    """``state.json`` of one run: its HOME, and per step status, inputs and outputs.

    A step is ``running`` while it works, ``done`` with the SHA-256 of its
    inputs and its outputs, or ``failed`` with the reason.  The next run skips
    a ``done`` step whose inputs and output files are unchanged, continues a
    ``running`` one (training resumes from its last checkpoint), and stops at
    a ``failed`` one whose inputs did not change (fix the cause first).
    """

    def __init__(self, directory: Path) -> None:
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "logs").mkdir(exist_ok=True)
        self.path = self.dir / "state.json"
        self.data: dict[str, Any] = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.data.setdefault("steps", {})
        self._lock_file = None

    def lock(self) -> None:
        self._lock_file = open(self.dir / ".lock", "w")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PipelineError(f"another retrain_all_for_home.py holds {self.dir}", EXIT_BUSY) from error

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True, default=str))
        os.replace(tmp, self.path)

    def log(self, message: str) -> None:
        line = f"{datetime.now():%F %T} {message}"
        print(line, flush=True)
        with open(self.dir / "STATUS.log", "a") as stream:
            stream.write(line + "\n")

    def step(self, name: str) -> dict[str, Any]:
        return self.data["steps"].setdefault(name, {})

    def outputs_intact(self, name: str) -> bool:
        record = self.data["steps"].get(name, {})
        for path, digest in (record.get("files") or {}).items():
            file = Path(path)
            if not file.is_file() or sha256(file) != digest:
                return False
        return True


def record_files(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path): sha256(path) for path in paths}


class Jobs:
    """Runs the pipeline's subprocesses, each in its own process group."""

    def __init__(self, state: State, *, stall_minutes: dict[str, float], wait_for_gpu: bool,
                 external_min_envs: int) -> None:
        self.state = state
        self.stall_s = {kind: 60.0 * minutes for kind, minutes in stall_minutes.items()}
        self.wait_for_gpu = wait_for_gpu
        self.external_min_envs = external_min_envs
        self.children: dict[int, str] = {}

    # -- external GPU training -------------------------------------------------
    def external_training(self) -> list[str]:
        """Command lines of GPU processes training something that is not ours."""

        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=60, check=False,
            ).stdout
        except (OSError, subprocess.TimeoutExpired):
            return []
        found = []
        for token in out.split():
            if not token.strip().isdigit():
                continue
            pid = int(token)
            try:
                group = os.getpgid(pid)
                cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
            except (OSError, ProcessLookupError):
                continue
            if group in self.children or pid == os.getpid():
                continue
            if is_training_command(cmdline, self.external_min_envs):
                found.append(f"{pid}: {cmdline[:160]}")
        return found

    def wait_for_free_gpu(self, name: str) -> None:
        if not self.wait_for_gpu:
            return
        started, noted = time.time(), -1.0
        while True:
            others = self.external_training()
            if not others:
                waited = time.time() - started
                if waited > 60:
                    self.state.log(f"{name}: waited {waited / 60:.0f} min for another training on the GPU")
                return
            if noted < 0 or time.time() - noted > 1800:
                self.state.log(f"{name}: waiting for another training on the GPU to end: {others}")
                noted = time.time()
            time.sleep(60)

    # -- jobs --------------------------------------------------------------------
    def start(self, name: str, cmd: list[str], *, env: dict[str, str] | None = None,
              cwd: Path = REPO) -> tuple[subprocess.Popen, Path]:
        log = self.state.dir / "logs" / f"{datetime.now():%m%d-%H%M%S}_{name}.log"
        self.state.log(f"start {name}: {' '.join(cmd)}  (log {log.name})")
        stream = open(log, "w")
        proc = subprocess.Popen(cmd, cwd=cwd, env=dict(os.environ, **(env or {})), stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)
        stream.close()
        self.children[proc.pid] = name
        return proc, log

    def run(self, name: str, cmd: list[str], kind: str, *, env: dict[str, str] | None = None,
            cwd: Path = REPO, gpu: bool = True, check: bool = True,
            poll: Callable[[], None] | None = None,
            on_start: Callable[[Path], None] | None = None) -> tuple[int, Path]:
        """Run one job to its end (stall detection, optional ``poll`` every 30 s).

        An exception raised by ``poll`` stops the job (its process group) and
        propagates.
        """

        if gpu:
            self.wait_for_free_gpu(name)
        started = time.time()
        proc, log = self.start(name, cmd, env=env, cwd=cwd)
        if on_start is not None:
            on_start(log)
        try:
            while True:
                try:
                    proc.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if time.time() - log.stat().st_mtime > self.stall_s[kind]:
                    self.kill(proc)
                    raise PipelineError(
                        f"STALL {name}: no output for {self.stall_s[kind] / 60:.0f} min, stopped (log {log})",
                        EXIT_STALL)
                if poll is not None:
                    poll()
            if poll is not None:
                poll()
        except BaseException:
            self.kill(proc)
            raise
        finally:
            self.children.pop(proc.pid, None)
        minutes = (time.time() - started) / 60
        if proc.returncode != 0 and check:
            tail = log.read_text(errors="replace")[-1500:].strip()
            raise PipelineError(f"{name} failed rc={proc.returncode} after {minutes:.1f} min (log {log}):\n{tail}")
        self.state.log(f"done {name} rc={proc.returncode} in {minutes:.1f} min")
        return proc.returncode, log

    def kill(self, proc: subprocess.Popen) -> None:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=60)
                return
            except subprocess.TimeoutExpired:
                continue

    def kill_all(self) -> None:
        for pid in list(self.children):
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def is_training_command(cmdline: str, min_envs: int) -> bool:
    """A training run: mjlab's ``train`` entry point, or >= ``min_envs`` environments."""

    words = cmdline.split()
    if any(Path(word).name == "train" for word in words):
        return True
    match = re.search(r"--env\.scene\.num-envs[ =](\d+)", cmdline)
    return bool(match) and int(match.group(1)) >= min_envs


def capture(cmd: list[str], *, cwd: Path = REPO, env: dict[str, str] | None = None,
            timeout: float = 1800, check: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, cwd=cwd, env=dict(os.environ, **(env or {})), capture_output=True,
                            text=True, timeout=timeout)
    if check and result.returncode != 0:
        raise PipelineError(f"{' '.join(cmd)} failed rc={result.returncode}: "
                            f"{(result.stderr or result.stdout).strip()[-1500:]}")
    return result


def git(repo: Path, *args: str, check: bool = True) -> str:
    return capture(["git", "-C", str(repo), *args], check=check).stdout.rstrip("\n")


def last_json_line(text: str, prefix: str = "") -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if prefix and line.startswith(prefix):
            return json.loads(line[len(prefix):])
        if not prefix and line.startswith("{"):
            return json.loads(line)
    raise PipelineError(f"no {'RESULT ' if prefix else 'JSON '}line in the output")


def python_cmd() -> list[str]:
    """The interpreter of this process when it is the project's own, else ``uv run``."""

    return [sys.executable] if Path(sys.prefix).resolve() == (REPO / ".venv").resolve() else [*UV, "python"]
