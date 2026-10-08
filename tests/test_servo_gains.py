"""Every learned policy trains with one servo gain on all 21 joints.

The robot runs walking, get-up and PICO at its KP_RL (125) on every servo,
head and neck included; the gain is defined once (SERVO_KP_POLICY).
"""

from __future__ import annotations

import unittest

from bam.mjlab import BamActuatorCfg

from mjlab_microban.robot.microban_constants import MICROBAN_ROBOT_CFG, SERVO_KP_POLICY
from mjlab_microban.tasks.microban_getup_env_cfg import make_microban_getup_env_cfg
from mjlab_microban.tasks.microban_teleop_v12_hand_pose_release import (
    make_microban_teleop_v12_hand_pose_release_env_cfg,
)
from mjlab_microban.tasks.microban_velocity_env_cfg import make_microban_velocity_env_cfg

# microban src/constants.py KP_RL.
ROBOT_KP_RL = 125


class ServoGainTest(unittest.TestCase):
    def test_one_gain_for_every_policy_and_joint(self) -> None:
        self.assertEqual(SERVO_KP_POLICY, ROBOT_KP_RL)
        for name, cfg in (
            ("walk", make_microban_velocity_env_cfg()),
            ("getup", make_microban_getup_env_cfg()),
            ("pico", make_microban_teleop_v12_hand_pose_release_env_cfg()),
        ):
            with self.subTest(name):
                robot = cfg.scene.entities["robot"]
                actuators = robot.articulation.actuators
                self.assertEqual(len(actuators), 1)
                self.assertIs(type(actuators[0]), BamActuatorCfg)
                self.assertEqual(tuple(actuators[0].target_names_expr), (r".*",))
                self.assertEqual(actuators[0].kp_fw, SERVO_KP_POLICY)
        self.assertEqual(MICROBAN_ROBOT_CFG.articulation.actuators[0].kp_fw, SERVO_KP_POLICY)


if __name__ == "__main__":
    unittest.main()
