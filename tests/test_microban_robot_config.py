# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Regression tests for the Microban robot simulation configuration."""

from __future__ import annotations

import re
import unittest

from mjlab_microban.robot.microban_constants import MICROBAN_ROBOT_CFG


class MicrobanRobotConfigTest(unittest.TestCase):
    def test_all_numbered_foot_collision_geoms_use_3d_priority_contacts(self) -> None:
        spec = MICROBAN_ROBOT_CFG.build().spec
        expected_names = {
            f"{side}_foot_collision_{index}"
            for side in ("left", "right")
            for index in range(1, 7)
        }
        numbered_foot_collision_names = {
            geom.name
            for geom in spec.geoms
            if re.fullmatch(r"(?:left|right)_foot_collision_[1-6]", geom.name)
        }
        self.assertEqual(numbered_foot_collision_names, expected_names)

        for name in sorted(expected_names):
            with self.subTest(geom=name):
                geom = spec.geom(name)
                self.assertEqual(geom.condim, 3)
                self.assertEqual(geom.priority, 1)


if __name__ == "__main__":
    unittest.main()
