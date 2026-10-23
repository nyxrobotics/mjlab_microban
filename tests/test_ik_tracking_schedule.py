"""One IK tracking run: the schedule table, the adapter columns and the packaged ranges.

The curriculum stages, the actor's trainable columns, the evaluators' profile
clock and the packager all read mjlab_microban/schedules.py.  The package's
foot-target ranges are those the policy trained with after the last stage
(the play/export env has no curriculum).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from mjlab_microban.schedules import (
    IK_TRACKING_ADAPTER_SCHEDULE_REVISION,
    IK_TRACKING_CRITIC_WARMUP,
    IK_TRACKING_SCHEDULE,
    IK_TRACKING_STEPS_PER_UPDATE,
    IK_TRACKING_TOTAL_UPDATES,
)
from mjlab_microban import policy_contract as contract
from mjlab_microban.tasks.curriculum import apply_to_cfg, final_settings
from mjlab_microban.tasks.microban_ik_tracking_env_cfg import IK_TRACKING_STAGES
from mjlab_microban.tasks.microban_ik_tracking_v12_actor import (
    IK_TRACKING_V12_ARM_OBSERVATION_COLUMNS,
    IK_TRACKING_V12_EXTRA_OBSERVATION_COLUMNS,
    IK_TRACKING_V12_HMD_OBSERVATION_COLUMNS,
    ik_tracking_v12_active_adapter_columns,
)
from mjlab_microban.tasks.microban_ik_tracking_v12_env_cfg import MicrobanIkTrackingV12RlCfg
from mjlab_microban.tasks.microban_ik_tracking_mdp import IK_TRACKING_ARM_HOME_PROBABILITY
from mjlab_microban.tasks.microban_ik_tracking_v13_arm_overlay import (
    make_microban_ik_tracking_v13_arm_overlay_env_cfg,
)

REPO = Path(__file__).resolve().parents[1]


class IkTrackingScheduleTest(unittest.TestCase):
    def test_the_table(self) -> None:
        self.assertEqual(IK_TRACKING_CRITIC_WARMUP, 1000)
        self.assertEqual(
            IK_TRACKING_SCHEDULE, {"arm": 1000, "foot": 4000, "foot_tighten": 6000}
        )
        self.assertEqual(IK_TRACKING_TOTAL_UPDATES, 15000)
        self.assertEqual(MicrobanIkTrackingV12RlCfg.max_iterations, IK_TRACKING_TOTAL_UPDATES)
        self.assertEqual(MicrobanIkTrackingV12RlCfg.num_steps_per_env, IK_TRACKING_STEPS_PER_UPDATE)
        self.assertEqual(IK_TRACKING_ADAPTER_SCHEDULE_REVISION, "freeze_extra_residual_to1000_then_hmd_arm_residual_to4000_then_all_v4")

    def test_curriculum_stages_switch_at_the_table(self) -> None:
        self.assertEqual(
            [stage.iteration for stage in IK_TRACKING_STAGES],
            [IK_TRACKING_SCHEDULE[name] for name in ("arm", "foot", "foot_tighten")],
        )

    def test_adapter_columns_open_with_the_stages(self) -> None:
        steps = IK_TRACKING_STEPS_PER_UPDATE
        arm, foot = IK_TRACKING_SCHEDULE["arm"], IK_TRACKING_SCHEDULE["foot"]
        self.assertEqual(ik_tracking_v12_active_adapter_columns(arm * steps), ())
        self.assertEqual(
            ik_tracking_v12_active_adapter_columns(arm * steps + steps),
            (*IK_TRACKING_V12_HMD_OBSERVATION_COLUMNS, *IK_TRACKING_V12_ARM_OBSERVATION_COLUMNS),
        )
        self.assertEqual(
            ik_tracking_v12_active_adapter_columns(foot * steps + steps), IK_TRACKING_V12_EXTRA_OBSERVATION_COLUMNS
        )

    def test_packaged_foot_ranges_are_the_trained_final_ranges(self) -> None:
        cfg = make_microban_ik_tracking_v13_arm_overlay_env_cfg()
        apply_to_cfg(cfg, final_settings(IK_TRACKING_STAGES))
        foot = cfg.commands["foot_target"]
        reach, lift = foot.reach_xy_range, foot.lift_height_range
        self.assertEqual(contract.IK_TRACKING_FOOT_TARGET_LOWER, [reach[0], reach[0], 0.0] * 2)
        self.assertEqual(contract.IK_TRACKING_FOOT_TARGET_UPPER, [reach[1], reach[1], lift[1]] * 2)
        both_reach, both_lift = foot.both_feet_reach_xy_range, foot.both_feet_lift_height_range
        self.assertEqual(contract.IK_TRACKING_BOTH_FEET_TARGET_LOWER, [both_reach[0], both_reach[0], 0.0] * 2)
        self.assertEqual(contract.IK_TRACKING_BOTH_FEET_TARGET_UPPER, [both_reach[1], both_reach[1], both_lift[1]] * 2)
        self.assertNotIn("hand_target", cfg.commands)
        self.assertEqual(
            cfg.events["ik_tracking_arm_target_motion"].params["home_probability"], IK_TRACKING_ARM_HOME_PROBABILITY
        )

    def test_a_dry_run_scales_every_switch_in_order(self) -> None:
        code = ("import json; from mjlab_microban import schedules as s; "
                "print(json.dumps([s.IK_TRACKING_SCHEDULE, s.IK_TRACKING_TOTAL_UPDATES]))")
        env = dict(os.environ, MICROBAN_SCHEDULE_SCALE="0.001", PYTHONPATH=str(REPO / "src"))
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        schedule, total = json.loads(out.stdout)
        values = [schedule[name] for name in ("arm", "foot", "foot_tighten")] + [total]
        self.assertEqual(values, sorted(set(values)))


if __name__ == "__main__":
    unittest.main()
