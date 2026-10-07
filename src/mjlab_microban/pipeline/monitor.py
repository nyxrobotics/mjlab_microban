"""Watching a training run: its log, its curriculum stages, checks and early stop.

``Monitor.poll`` runs every 30 s while a training job runs (and once after
it ends).  It

* reads the new lines of the training log: the update counter, the metrics
  the rules need, and every ``Curriculum stage ... (update U)`` line, which
  must name a stage of the task's table (WALK_STAGES / TELEOP_STAGES /
  GETUP_STAGES, read by the pipeline) and come within a tolerance of its
  update; once the run ends (``require_stages``) every stage due by its last
  checkpoint must have been seen;
* starts one check at a time on saved checkpoints, oldest first (a separate
  process, in parallel with training; every due checkpoint is checked, also
  after the training ended), and records its verdict in state.json;
* stops the training early once two consecutive checks (N and N+1000) pass
  and N+1000 is at or after the earliest end of the run: the run then ends
  with checkpoint N+1000 (``EarlyStop``), which is judged once.

A failing check never stops a run: the run trains its planned updates and
the pipeline goes on with its best checkpoint (user, 2026-10-07: no stop on
a failed test, only on a broken program).  Intervals: config/pipeline.yaml
(``<step>.check``); docs in docs/home_pose_workflow.md.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from mjlab_microban.pipeline.core import PipelineError, find_checkpoint

STAGE_LINE = re.compile(r"Curriculum stage (\d+) (.+?) at step (\d+) \(update (\d+)\)")
ITERATION_LINE = re.compile(r"Learning iteration (\d+)/(\d+)")


class EarlyStop(Exception):
    """Two consecutive checks passed: the run ends with ``checkpoint``."""

    def __init__(self, checkpoint: Path) -> None:
        super().__init__(str(checkpoint))
        self.checkpoint = checkpoint


@dataclass
class Check:
    """One running check: the process, its log, and how to read the verdict."""

    update: int
    process: Any
    log: Path
    verdict: Callable[[Path], dict[str, Any]]


@dataclass
class Monitor:
    """See the module doc.  ``record`` is the step's dict in state.json."""

    state: Any  # core.State
    jobs: Any  # core.Jobs
    record: dict[str, Any]
    experiment: str
    label: str
    expected_stages: dict[str, int]  # stage name -> update it must start at
    stage_tolerance: int
    check_updates: list[int]  # checkpoints (model_<update>) to check, in order
    min_final_update: int  # earliest model_<update> a run may end with
    start_check: Callable[[Path], Check] | None = None
    early_stop: bool = True
    log_path: Path | None = None
    resumed_from: int = 0  # the update a continued run started from (stages re-applied there)
    _offset: int = 0
    _iteration: int = -1
    _running: Check | None = None

    def __post_init__(self) -> None:
        self.record.setdefault("checks", {})
        self.record.setdefault("stages_seen", {})

    # -- the training log ------------------------------------------------------------
    def read_log(self) -> None:
        if self.log_path is None or not self.log_path.is_file():
            return
        with open(self.log_path, errors="replace") as stream:
            stream.seek(self._offset)
            text = stream.read()
            self._offset = stream.tell()
        for line in text.splitlines():
            line = re.sub(r"\x1b\[[0-9;]*m", "", line)
            match = ITERATION_LINE.search(line)
            if match:
                self._iteration = int(match.group(1))
                continue
            match = STAGE_LINE.search(line)
            if match:
                self.stage_seen(match.group(2), int(match.group(4)))

    def stage_seen(self, name: str, update: int) -> None:
        if name in self.record["stages_seen"]:
            return  # re-applied by a continued run
        self.record["stages_seen"][name] = update
        expected = self.expected_stages.get(name)
        if expected is None:
            raise PipelineError(f"curriculum stage {name!r} (update {update}) is not in the schedule table "
                                f"{sorted(self.expected_stages)}: the run trains another schedule")
        latest = max(expected, self.resumed_from) + self.stage_tolerance
        if not expected <= update <= latest:
            raise PipelineError(f"curriculum stage {name!r} applied at update {update}, the table says "
                                f"{expected} (tolerance +{self.stage_tolerance}): the schedule did not run as written")
        self.state.log(f"stage {name} at update {update} (table {expected})")

    def require_stages(self, final_update: int) -> None:
        """Every stage the table starts by ``final_update`` (plus the tolerance) was applied."""

        missing = sorted(name for name, update in self.expected_stages.items()
                         if update + self.stage_tolerance <= final_update
                         and name not in self.record["stages_seen"])
        if missing:
            raise PipelineError(f"the run reached update {final_update} without applying the curriculum "
                                f"stages {missing}: the schedule did not run as written")

    # -- checks ------------------------------------------------------------------------
    def due_checks(self) -> list[int]:
        done = {int(k) for k in self.record["checks"]}
        return [u for u in self.check_updates if u not in done
                and find_checkpoint(self.experiment, self.label, u) is not None]

    def collect(self) -> None:
        running = self._running
        if running is None or running.process.poll() is None:
            return
        self._running = None
        self.jobs.children.pop(running.process.pid, None)
        try:
            verdict = running.verdict(running.log)
        except Exception as error:  # noqa: BLE001 - an evaluator that died is not a verdict
            verdict = {"passed": False, "error": f"{type(error).__name__}: {error}"[:300]}
        self.record["checks"][str(running.update)] = verdict
        self.state.save()
        self.state.log(f"check model_{running.update}: {'PASS' if verdict.get('passed') else 'FAIL'} "
                       f"{json.dumps(verdict.get('summary', verdict.get('error', '')), default=str)[:600]}")
        self.decide_early_stop()

    def decide_early_stop(self) -> None:
        if not self.early_stop:
            return
        checks = self.record["checks"]
        for update in self.check_updates:
            previous = update - 1000
            if update < self.min_final_update or str(previous) not in checks or str(update) not in checks:
                continue
            if checks[str(previous)].get("passed") and checks[str(update)].get("passed"):
                checkpoint = find_checkpoint(self.experiment, self.label, update)
                if checkpoint is not None:
                    self.record["adopted"] = str(checkpoint)
                    self.state.save()
                    self.state.log(f"checks of model_{previous} and model_{update} passed: the run ends with "
                                   f"model_{update}")
                    raise EarlyStop(checkpoint)

    def start_due(self) -> None:
        if self.start_check is None or self._running is not None:
            return
        due = self.due_checks()
        if not due:
            return
        update = due[0]  # oldest first: every checkpoint is checked (the best one is chosen from them)
        checkpoint = find_checkpoint(self.experiment, self.label, update)
        self._running = self.start_check(checkpoint)
        self._running.update = update
        self.state.save()

    def poll(self) -> None:
        self.read_log()
        self.collect()
        self.start_due()

    def finish(self) -> None:
        """After training ended: run every check still due, one at a time, deciding after each."""

        while True:
            while self._running is not None:
                try:
                    self._running.process.wait(timeout=30)
                except Exception:  # noqa: BLE001 - subprocess.TimeoutExpired
                    pass
                self.collect()
            self.start_due()
            if self._running is None:
                return

    def stop(self) -> None:
        if self._running is not None:
            self.jobs.kill(self._running.process)
            self.jobs.children.pop(self._running.process.pid, None)
            self._running = None

