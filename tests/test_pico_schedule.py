"""One PICO run: the schedule table and the adapter columns.

The curriculum stages, the actor's trainable columns and the evaluators'
profile clock all read mjlab_microban/schedules.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from mjlab_microban.schedules import (
    PICO_ADAPTER_SCHEDULE_REVISION,
    PICO_CRITIC_WARMUP,
    PICO_SCHEDULE,
    PICO_STEPS_PER_UPDATE,
    PICO_TOTAL_UPDATES,
)
from mjlab_microban.tasks.microban_teleop_env_cfg import TELEOP_STAGES
from mjlab_microban.tasks.microban_teleop_v12_actor import (
    TELEOP_V12_EXTRA_OBSERVATION_COLUMNS,
    TELEOP_V12_HAND_OBSERVATION_COLUMNS,
    TELEOP_V12_HMD_OBSERVATION_COLUMNS,
    teleop_v12_active_adapter_columns,
)
from mjlab_microban.tasks.microban_teleop_v12_env_cfg import MicrobanTeleopV12RlCfg

REPO = Path(__file__).resolve().parents[1]


class PicoScheduleTest(unittest.TestCase):
    def test_the_table(self) -> None:
        self.assertEqual(PICO_CRITIC_WARMUP, 1000)
        self.assertEqual(
            PICO_SCHEDULE, {"hand": 1000, "hand_tighten": 2500, "foot": 4000, "foot_tighten": 6000}
        )
        self.assertEqual(PICO_TOTAL_UPDATES, 9000)
        self.assertEqual(MicrobanTeleopV12RlCfg.max_iterations, PICO_TOTAL_UPDATES)
        self.assertEqual(MicrobanTeleopV12RlCfg.num_steps_per_env, PICO_STEPS_PER_UPDATE)
        self.assertEqual(PICO_ADAPTER_SCHEDULE_REVISION, "freeze_extra_to1000_then_hmd_hand_to4000_then_all_v2")

    def test_curriculum_stages_switch_at_the_table(self) -> None:
        self.assertEqual(
            [stage.iteration for stage in TELEOP_STAGES],
            [PICO_SCHEDULE[name] for name in ("hand", "hand_tighten", "foot", "foot_tighten")],
        )

    def test_adapter_columns_open_with_the_stages(self) -> None:
        steps = PICO_STEPS_PER_UPDATE
        hand, foot = PICO_SCHEDULE["hand"], PICO_SCHEDULE["foot"]
        self.assertEqual(teleop_v12_active_adapter_columns(hand * steps), ())
        self.assertEqual(
            teleop_v12_active_adapter_columns(hand * steps + steps),
            (*TELEOP_V12_HMD_OBSERVATION_COLUMNS, *TELEOP_V12_HAND_OBSERVATION_COLUMNS),
        )
        self.assertEqual(
            teleop_v12_active_adapter_columns(foot * steps + steps), TELEOP_V12_EXTRA_OBSERVATION_COLUMNS
        )

    def test_a_dry_run_scales_every_switch_in_order(self) -> None:
        code = ("import json; from mjlab_microban import schedules as s; "
                "print(json.dumps([s.PICO_SCHEDULE, s.PICO_TOTAL_UPDATES]))")
        env = dict(os.environ, MICROBAN_SCHEDULE_SCALE="0.001", PYTHONPATH=str(REPO / "src"))
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        schedule, total = json.loads(out.stdout)
        values = [schedule[name] for name in ("hand", "hand_tighten", "foot", "foot_tighten")] + [total]
        self.assertEqual(values, sorted(set(values)))


if __name__ == "__main__":
    unittest.main()
