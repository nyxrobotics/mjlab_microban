"""The one contract between the trained policies and the robot runtime.

Every exported ONNX (walk, get-up, PICO) carries these metadata keys, and a
release has one ``manifest.json`` listing the three files (docs/policies.md):

* ``policy_contract``: POLICY_CONTRACT, bumped whenever the meaning of any
  input, output or metadata key changes.  The robot accepts exactly the
  version it implements; it does not hash its own sources into the package
  any more (each release is checked by the robot's validator and its startup
  self-test on the packaged smoke observations instead).
* ``servo_kp``: the firmware P gain the policy was trained at on every servo
  (robot/microban_constants.SERVO_KP_POLICY); the robot runs the policy at
  exactly this gain.
* ``home_tag`` / ``home_joint_hash``: the HOME of config/home_pose.yaml the
  policy was trained at (the robot reads the same YAML).

The manifest records each file's SHA-256 next to its checkpoint SHA-256, the
judgments the pipeline made, the PICO schedule and the training commit.
"""

from __future__ import annotations

from mjlab_microban.robot.home_pose import HOME
from mjlab_microban.robot.microban_constants import SERVO_KP_POLICY

POLICY_CONTRACT = "microban-policy-1"
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
POLICY_FILES = {
    "walk": "walk.onnx",
    "getup": "getup.onnx",
    "pico_teleop": "pico_teleop.onnx",
}


def contract_metadata() -> dict[str, str]:
    """The contract keys every exported policy carries."""

    return {
        "policy_contract": POLICY_CONTRACT,
        "servo_kp": str(SERVO_KP_POLICY),
        "home_tag": HOME.tag,
        "home_joint_hash": HOME.joint_hash,
    }
