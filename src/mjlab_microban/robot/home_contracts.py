"""Every HOME-bound contract, recipe and revision string, in one place.

A string that names the HOME a checkpoint, gate or ONNX was trained at is
computed here from the loaded HOME (``config/home_pose.yaml``):

* the HOME with published artifacts (forward-lean, trunk +10 deg) keeps its
  published strings exactly (``home_pose.PUBLISHED_HOME_OVERRIDES[...]
  ["contracts"]``), so its checkpoints, gates and packages stay valid;
* any other HOME gets a derived string that embeds ``HOME.tag``
  (``<label>_<10-digit joint hash>``), so artifacts of another HOME are
  refused automatically.

The derived strings follow the mechanisms the HOME uses: a vertical trunk
(``trunk_pitch_deg == 0``) trains with the trunk-frame targets and the
original hand-FK box, a pitched trunk with the HOME-levelled targets, the
level-headset HMD neutral and the receiver-capped hand box.

Nothing here imports mjlab; the training modules import these constants.
"""

from __future__ import annotations

from mjlab_microban.robot.home_pose import HOME, signed_degree_token
from mjlab_microban.schedules import PICO_CRITIC_WARMUP, PICO_TOTAL_UPDATES

_TAG = HOME.tag
_UPRIGHT = HOME.trunk_is_vertical


def _contract(key: str, upright: str, pitched: str) -> str:
    return HOME.contract(key, upright if _UPRIGHT else pitched)


_HIP = signed_degree_token(HOME.hip_pitch_deg)
_ANKLE = signed_degree_token(HOME.ankle_pitch_deg)
_V12_RECIPE_BASE = "velocity_source_staged_mask_reachable_fk_elbow_minus10_raw_prev_action_servo_range_pi"
_V12_LEVELLED = "home_levelled_targets_level_hmd_receiver_box_hands"

# PICO contract v12 checkpoints, gates and packages (microban_teleop_v12_env_cfg).
V12_HOME_POSE_REVISION = _contract(
    "v12_home_pose_revision",
    f"{_TAG}_hip_{_HIP}_ankle_{_ANKLE}_shoulder_zero_v5",
    f"{_TAG}_hip_{_HIP}_ankle_{_ANKLE}_shoulder_zero_v6",
)
V12_RECIPE_REVISION = _contract(
    "v12_recipe_revision",
    f"{_TAG}_{_V12_RECIPE_BASE}_v11",
    f"{_TAG}_{_V12_RECIPE_BASE}_{_V12_LEVELLED}_v17",
)
# The PICO recipe every run trains: the active-hand arm pose release and one
# run with a critic warm-up (mjlab_microban/schedules.py).
V12_HAND_POSE_RELEASE_RECIPE_REVISION = (
    f"{_TAG}_{_V12_RECIPE_BASE}{'' if _UPRIGHT else '_' + _V12_LEVELLED}"
    f"_active_hand_arm_pose_release_one_run_warmup{PICO_CRITIC_WARMUP}"
    f"_total{PICO_TOTAL_UPDATES}_v1"
)


def contract_strings() -> dict[str, str]:
    """The HOME-bound identifiers stamped into this HOME's checkpoints and gates."""

    return {
        "v12_home_pose_revision": V12_HOME_POSE_REVISION,
        "v12_recipe_revision": V12_RECIPE_REVISION,
        "v12_hand_pose_release_recipe_revision": V12_HAND_POSE_RELEASE_RECIPE_REVISION,
        # Frame of the PICO foot/hand target columns (microban_hand_fk).
        "v12_target_frame": _target_frame(),
    }


def _target_frame() -> str:
    from mjlab_microban.robot.microban_hand_fk import MICROBAN_HAND_TARGET_FRAME

    return MICROBAN_HAND_TARGET_FRAME
