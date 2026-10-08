"""Watching a training run: its curriculum stages against the schedule table.

``Monitor.poll`` runs every 30 s while a training job runs (and once after
it ends).  It reads the new lines of the training log, and every
``Curriculum stage ... (update U)`` line must name a stage of the task's
table (WALK_STAGES / TELEOP_STAGES / GETUP_STAGES, read by the pipeline) and
come within a tolerance of its update; once the run ends
(``require_stages``) every stage due by its last checkpoint must have been
seen.  The tolerance is config/pipeline.yaml ``<step>.stage_tolerance``;
docs in docs/home_pose_workflow.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mjlab_microban.pipeline.core import PipelineError

STAGE_LINE = re.compile(r"Curriculum stage (\d+) (.+?) at step (\d+) \(update (\d+)\)")


@dataclass
class Monitor:
    """See the module doc.  ``record`` is the step's dict in state.json."""

    state: Any  # core.State
    record: dict[str, Any]
    expected_stages: dict[str, int]  # stage name -> update it must start at
    stage_tolerance: int
    log_path: Path | None = None
    _offset: int = 0

    def __post_init__(self) -> None:
        self.record.setdefault("stages_seen", {})

    def read_log(self) -> None:
        if self.log_path is None or not self.log_path.is_file():
            return
        with open(self.log_path, errors="replace") as stream:
            stream.seek(self._offset)
            text = stream.read()
            self._offset = stream.tell()
        for line in text.splitlines():
            match = STAGE_LINE.search(re.sub(r"\x1b\[[0-9;]*m", "", line))
            if match:
                self.stage_seen(match.group(2), int(match.group(4)))

    def stage_seen(self, name: str, update: int) -> None:
        if name in self.record["stages_seen"]:
            return
        self.record["stages_seen"][name] = update
        expected = self.expected_stages.get(name)
        if expected is None:
            raise PipelineError(f"curriculum stage {name!r} (update {update}) is not in the schedule table "
                                f"{sorted(self.expected_stages)}: the run trains another schedule")
        if not expected <= update <= expected + self.stage_tolerance:
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

    def poll(self) -> None:
        self.read_log()
