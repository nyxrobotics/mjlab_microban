"""Can this training line retrain at a given HOME?

Some HOME edits are valid poses (``home_pose.py`` loads them) but the
training tasks of this checkout still refuse them, for example:

* ``trunk_pitch_deg`` other than 0: the upright observation/reward frames of
  this line (the forward-lean frames live on branch ``forward-lean-v2``);
* shoulder pitch other than 0: the PICO contract v12 HOME revision;
* an arm HOME whose reachable hand box leaves the PICO receiver's
  runtime-validated +-0.064 m box (``microban_hand_fk``).

Instead of a hand-kept list of such rules, ``check_training_line`` asks the
tasks themselves: a fresh Python process installs the candidate HOME as
``mjlab_microban.robot.home_pose.HOME`` and imports ``mjlab_microban.tasks``,
which builds the env configs of every registered Microban task (walking,
get-up stages, PICO v12 and its rescue stages).  If that import fails, none of
them can be retrained at the HOME, and the first error says why.  It takes a
few seconds (torch + mjlab import) and needs no GPU.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from mjlab_microban.robot.home_pose import REPO_ROOT

CHECK_TIMEOUT_S = 600.0


@dataclass(frozen=True)
class TrainingLineCheck:
    ok: bool
    error: str | None = None

    def summary(self) -> dict[str, object]:
        return {"ok": self.ok, "error": self.error}


def check_training_line(
    joint_pos_deg: Mapping[str, float],
    trunk_pitch_deg: float,
    *,
    name: str = "",
    label: str = "home",
    path: Path | str = "candidate HOME",
    timeout_s: float = CHECK_TIMEOUT_S,
) -> TrainingLineCheck:
    """Import the training tasks of this checkout at the given HOME (subprocess)."""

    request = json.dumps(
        {
            "joint_pos_deg": {key: float(value) for key, value in joint_pos_deg.items()},
            "trunk_pitch_deg": float(trunk_pitch_deg),
            "name": str(name),
            "label": str(label),
            "path": str(path),
        }
    )
    source = str(REPO_ROOT / "src")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [source, *filter(None, [environment.get("PYTHONPATH")])]
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "mjlab_microban.robot.home_pose_training"],
            input=request,
            capture_output=True,
            text=True,
            env=environment,
            cwd=str(REPO_ROOT),
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return TrainingLineCheck(False, f"the task import did not finish in {timeout_s:.0f} s")
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("{"):
            try:
                reply = json.loads(line)
            except json.JSONDecodeError:
                continue
            return TrainingLineCheck(bool(reply["ok"]), reply.get("error"))
    tail = (completed.stderr.strip().splitlines() or ["no output"])[-1]
    return TrainingLineCheck(False, f"the task import crashed (exit {completed.returncode}): {tail}")


def _child() -> int:
    request = json.load(sys.stdin)
    import warnings

    warnings.filterwarnings("ignore")
    from mjlab_microban.robot import home_pose

    error: str | None = None
    try:
        home = home_pose.home_pose_from_values(
            joint_pos_deg=request["joint_pos_deg"],
            trunk_pitch_deg=request["trunk_pitch_deg"],
            name=request["name"],
            label=request["label"],
            path=request["path"],
        )
    except ValueError as exception:
        error = f"the HOME itself is invalid: {exception}"
    else:
        # Every consumer reads home_pose.HOME (lazily loaded); install the
        # candidate before anything imports it.
        home_pose.HOME = home
        captured = io.StringIO()
        try:
            # mjlab's plugin loader prints "[WARN] Failed to load task package"
            # for the same error; keep stdout for the JSON reply.
            with contextlib.redirect_stdout(captured):
                import mjlab_microban.tasks  # noqa: F401
        except Exception as exception:  # noqa: BLE001 - any task refusal
            message = " ".join(str(exception).split())
            error = f"{type(exception).__name__}: {message}"
    print(json.dumps({"ok": error is None, "error": error}))
    return 0


if __name__ == "__main__":
    raise SystemExit(_child())
