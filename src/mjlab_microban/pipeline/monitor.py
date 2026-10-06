"""Watching a training run: its log, its curriculum stages, checks and early stop.

``Monitor.poll`` runs every 30 s while a training job runs (and once after
it ends).  It

* reads the new lines of the training log: the update counter, the metrics
  the rules need, and every ``Curriculum stage ... (update U)`` line, which
  must come within a tolerance of the update the table (schedules.py) says;
* applies the policy's stop rules to the metrics (``abort`` -> the run stops
  with a report: no point in training on);
* starts one check at a time on saved checkpoints (a separate process,
  in parallel with training; a check that comes due while another runs is
  skipped and recorded), and records its verdict in state.json;
* stops the training early once two consecutive checks (N and N+1000) pass
  and N+1000 is at or after the earliest end of the run: the run then ends
  with checkpoint N+1000 (``EarlyStop``), which is judged once.

Rules and intervals: config/pipeline.yaml (``<step>.check``); docs in
docs/home_pose_workflow.md.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from mjlab_microban.pipeline.core import PipelineError, find_checkpoint

STAGE_LINE = re.compile(r"Curriculum stage (\d+) (.+?) at step (\d+) \(update (\d+)\)")
ITERATION_LINE = re.compile(r"Learning iteration (\d+)/(\d+)")
METRIC_LINE = re.compile(r"^\s*([A-Za-z_/ ]+[A-Za-z_]):\s*(-?[0-9.]+(?:e-?\d+)?)\s*$")


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
    abort_rules: Callable[[int, dict[str, list[tuple[int, float]]]], str | None] | None = None
    check_abort: Callable[[int, dict[str, Any]], str | None] | None = None
    early_stop: bool = True
    log_path: Path | None = None
    resumed_from: int = 0  # the update a continued run started from (stages re-applied there)
    _offset: int = 0
    _iteration: int = -1
    _metrics: dict[str, list[tuple[int, float]]] = field(default_factory=dict)
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
                continue
            match = METRIC_LINE.match(line)
            if match and self._iteration >= 0:
                self._metrics.setdefault(match.group(1).strip(), []).append(
                    (self._iteration, float(match.group(2))))

    def stage_seen(self, name: str, update: int) -> None:
        if name in self.record["stages_seen"]:
            return  # re-applied by a continued run
        self.record["stages_seen"][name] = update
        expected = self.expected_stages.get(name)
        if expected is None:
            return
        latest = max(expected, self.resumed_from) + self.stage_tolerance
        if not expected <= update <= latest:
            raise PipelineError(f"curriculum stage {name!r} applied at update {update}, the table says "
                                f"{expected} (tolerance +{self.stage_tolerance}): the schedule did not run as written")
        self.state.log(f"stage {name} at update {update} (table {expected})")

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
                       f"{json.dumps(verdict.get('summary', verdict.get('error', '')), default=str)[:400]}")
        if self.check_abort is not None:
            reason = self.check_abort(running.update, verdict)
            if reason:
                raise PipelineError(f"stopped at the check of model_{running.update}: {reason}")
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
        # The newest due checkpoint; older ones that were not reached in time
        # are skipped (one check at a time, never queued behind training).
        for update in due[:-1]:
            self.record["checks"][str(update)] = {"passed": False, "skipped": "the previous check still ran"}
            self.state.log(f"check model_{update}: skipped (the previous check still ran)")
        update = due[-1]
        checkpoint = find_checkpoint(self.experiment, self.label, update)
        self._running = self.start_check(checkpoint)
        self._running.update = update
        self.state.save()

    def poll(self) -> None:
        self.read_log()
        if self.abort_rules is not None:
            reason = self.abort_rules(self._iteration, self._metrics)
            if reason:
                raise PipelineError(f"stopped at update {self._iteration}: {reason}")
        self.collect()
        self.start_due()

    def finish(self) -> None:
        """After training ended: wait for the running check and decide once more."""

        while self._running is not None:
            try:
                self._running.process.wait(timeout=30)
            except Exception:  # noqa: BLE001 - subprocess.TimeoutExpired
                pass
            self.collect()
        self.start_due()
        while self._running is not None:
            try:
                self._running.process.wait(timeout=30)
            except Exception:  # noqa: BLE001
                pass
            self.collect()

    def stop(self) -> None:
        if self._running is not None:
            self.jobs.kill(self._running.process)
            self.jobs.children.pop(self._running.process.pid, None)
            self._running = None


def last_value_at(metrics: dict[str, list[tuple[int, float]]], name: str, update: int) -> float | None:
    """The metric's value logged at ``update`` (or the last one before it)."""

    values = [value for it, value in metrics.get(name, []) if it <= update]
    return values[-1] if values else None


def walk_abort_rules(rules: dict[str, Any]) -> Callable[[int, dict], str | None]:
    """Episode-length floors early in training (falling over to cash in speed)."""

    def check(iteration: int, metrics: dict) -> str | None:
        for update, floor in rules["episode_length_floor"]:
            if iteration >= update:
                value = last_value_at(metrics, "Mean episode length", update)
                if value is not None and value < floor:
                    return (f"mean episode length {value:.0f} < {floor} at update {update} (old reward: "
                            "about twice that): the walker is not learning to stay up")
        return None

    return check


def walk_check_abort(rules: dict[str, Any]) -> Callable[[int, dict], str | None]:
    """At the hopeless point: falls, a wide ratio angle or almost no speed."""

    def check(update: int, verdict: dict) -> str | None:
        if update != rules["hopeless_update"] or "probe" not in verdict:
            return None
        probe = verdict["probe"]
        if not probe["checks"]["W4"]:
            return f"W4 (falls) fails at update {update}"
        if probe["angle_deg"][0] > rules["hopeless_angle_deg"]:
            return f"ratio angle {probe['angle_deg'][0]:.0f} > {rules['hopeless_angle_deg']} deg at update {update}"
        if probe["speed"][0] < rules["hopeless_speed"]:
            return f"speed fraction {probe['speed'][0]:.2f} < {rules['hopeless_speed']} at update {update}"
        return None

    return check


def getup_abort_rules(rules: dict[str, Any], switches: list[int]) -> Callable[[int, dict], str | None]:
    """standing_bonus floors (the policy stopped standing)."""

    name = "Episode_Reward/standing_bonus"

    def check(iteration: int, metrics: dict) -> str | None:
        for update, floor in rules["standing_bonus_floor"]:
            if iteration >= update:
                value = last_value_at(metrics, name, update)
                if value is not None and value < floor:
                    return f"standing_bonus {value:.2f} < {floor} at update {update}"
        first = rules["sustained_after"]
        run, needed = 0, rules["sustained_updates"]
        for update, value in metrics.get(name, []):
            if update < first or any(s <= update < s + rules["switch_grace"] for s in switches):
                run = 0
                continue
            run = run + 1 if value < rules["sustained_floor"] else 0
            if run >= needed:
                return f"standing_bonus below {rules['sustained_floor']} for {needed} updates (to {update})"
        return None

    return check
