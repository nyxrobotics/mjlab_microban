"""Get-up-only XC330 model with the robot's body/neck gain split."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields

from bam.mjlab import BamActuator, BamActuatorCfg

from mjlab_microban.robot.microban_constants import MICROBAN_ROBOT_CFG


GETUP_NECK_JOINT_NAMES = ("head", "neck_roll", "neck_pitch")
GETUP_BODY_KP_FW = 125.0
GETUP_NECK_KP_FW = 400.0


class GetupBamActuator(BamActuator):
    """Keep a shared battery model while giving neck joints their runtime P gain."""

    def initialize(self, mj_model, model, data, device: str) -> None:
        super().initialize(mj_model, model, data, device)
        if self.cfg.kp_fw != GETUP_BODY_KP_FW or self.kp_scale is None:
            raise ValueError("Get-up BAM model requires body firmware P gain 125")
        neck_indices = [
            index for index, name in enumerate(self.target_names)
            if name in GETUP_NECK_JOINT_NAMES
        ]
        if len(neck_indices) != len(GETUP_NECK_JOINT_NAMES):
            raise ValueError("Get-up BAM model must include all three head/neck joints")
        self.kp_scale = self.kp_scale.repeat(1, len(self.target_names))
        self.kp_scale[:, neck_indices] = GETUP_NECK_KP_FW / GETUP_BODY_KP_FW
        self.default_kp_scale = self.kp_scale.clone()


@dataclass(kw_only=True)
class GetupBamActuatorCfg(BamActuatorCfg):
    def build(self, entity, target_ids: list[int], target_names: list[str]) -> GetupBamActuator:
        return GetupBamActuator(self, entity, target_ids, target_names)


def make_getup_robot_cfg():
    """Clone the robot model so other policies keep their actuator configuration."""
    robot_cfg = deepcopy(MICROBAN_ROBOT_CFG)
    articulation = robot_cfg.articulation
    if articulation is None or len(articulation.actuators) != 1:
        raise ValueError("Expected one XC330 BAM actuator group for get-up")
    original = articulation.actuators[0]
    if not isinstance(original, BamActuatorCfg):
        raise TypeError("Get-up requires the XC330 BAM actuator configuration")
    parameters = {field.name: getattr(original, field.name) for field in fields(original)}
    articulation.actuators = (GetupBamActuatorCfg(**parameters),)
    return robot_cfg
