"""Every HOME-bound contract, recipe and revision string, in one place.

A string that names the HOME a checkpoint, gate or ONNX was trained at is
computed here from the loaded HOME (``config/home_pose.yaml``):

* the two HOMEs with published artifacts (centered, trunk vertical; and
  forward-lean, trunk +10 deg) keep their historical strings exactly
  (``home_pose.LEGACY_HOME_OVERRIDES[...]["contracts"]``), so their
  checkpoints, gates and robot packages stay valid;
* any other HOME gets a derived string that embeds ``HOME.tag``
  (``<label>_<10-digit joint hash>``), so artifacts of another HOME are
  refused automatically.

The derived strings follow the mechanisms the HOME uses: a vertical trunk
(``trunk_pitch_deg == 0``) trains with the trunk-frame targets and the
original hand-FK box (the centered line's recipes), a pitched trunk with the
HOME-levelled targets, the level-headset HMD neutral and the receiver-capped
hand box (the forward-lean line's recipes).

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

# Walking ONNX (export_walk_onnx.CONTRACT_VERSION; robot walk.py).
WALK_CONTRACT_VERSION = _contract(
    "walk_contract_version", f"v3_{_TAG}_servo_range", f"v4_{_TAG}_servo_range"
)
# Get-up checkpoints and ONNX (microban_getup_runner; robot getup.py).
GETUP_CONTRACT_VERSION = _contract("getup_contract_version", f"v5_{_TAG}", f"v6_{_TAG}")
# The pre-bump stamp the centered line still accepts for v5 runs (None: none).
GETUP_LEGACY_STAMP: str | None = HOME.contract("getup_legacy_stamp", "") or None

# PICO contract v12 (microban_teleop_v12_env_cfg; robot pico_hybrid.py).
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
# The PICO recipe every run trains (stage C, 2026-10-07): the active-hand arm
# pose release, the twist-ratio velocity term and one run with a critic
# warm-up (mjlab_microban/schedules.py).  New at every HOME: the published
# strings (v12 / v18) named the segmented recipe of the old reward.
V12_HAND_POSE_RELEASE_RECIPE_REVISION = (
    f"{_TAG}_{_V12_RECIPE_BASE}{'' if _UPRIGHT else '_' + _V12_LEVELLED}"
    f"_active_hand_arm_pose_release_twist_ratio_one_run_warmup{PICO_CRITIC_WARMUP}"
    f"_total{PICO_TOTAL_UPDATES}_v1"
)
V12_PACKAGER_REVISION = f"microban_pico_packager_one_run_v1_{_TAG}_servo_range"

# Independent upright full-body teleop task.
UPRIGHT_FULLBODY_RECIPE_REVISION = _contract(
    "upright_fullbody_recipe_revision",
    f"physical_neutral_full_actor_from_scratch_raw83x18_{_TAG}_v4",
    f"physical_neutral_full_actor_from_scratch_raw83x18_home_levelled_targets_"
    f"receiver_box_hands_{_TAG}_v6",
)
UPRIGHT_FULLBODY_HOME_REVISION = _contract(
    "upright_fullbody_home_revision",
    f"{_TAG}_shoulder_zero_v5",
    f"{_TAG}_com_centered_shoulder_zero_v6",
)

# Walking checkpoints without the HOME stamp (saved before it existed).
ACCEPTS_UNSTAMPED_WALK_CHECKPOINTS = bool(HOME.override("accepts_unstamped_walk_checkpoints", False))


def contract_strings() -> dict[str, str]:
    """The HOME-bound identifiers stamped into this HOME's checkpoints and gates.

    Training-side only: the robot checks a policy's ``home_pose`` stamp and the
    one contract ``microban-policy-1`` (policy_contract.py), not these.
    """

    return {
        "walk_contract_version": WALK_CONTRACT_VERSION,
        "getup_contract_version": GETUP_CONTRACT_VERSION,
        # "" : the checkpoint stamp is not checked (the centered line's v5
        # exporter also published v4-stamped v5 runs).
        "getup_checkpoint_stamp": "" if GETUP_LEGACY_STAMP else GETUP_CONTRACT_VERSION,
        "v12_home_pose_revision": V12_HOME_POSE_REVISION,
        "v12_recipe_revision": V12_RECIPE_REVISION,
        "v12_hand_pose_release_recipe_revision": V12_HAND_POSE_RELEASE_RECIPE_REVISION,
        "v12_packager_revision": V12_PACKAGER_REVISION,
        # Frame of the PICO foot/hand target columns (microban_hand_fk).
        "v12_target_frame": _target_frame(),
    }


def _target_frame() -> str:
    from mjlab_microban.robot.microban_hand_fk import MICROBAN_HAND_TARGET_FRAME

    return MICROBAN_HAND_TARGET_FRAME
