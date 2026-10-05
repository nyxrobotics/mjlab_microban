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
V12_HAND_POSE_RELEASE_RECIPE_REVISION = _contract(
    "v12_hand_pose_release_recipe_revision",
    f"{_TAG}_{_V12_RECIPE_BASE}_active_hand_arm_pose_release_v12",
    f"{_TAG}_{_V12_RECIPE_BASE}_{_V12_LEVELLED}_active_hand_arm_pose_release_v18",
)
V12_PACKAGER_REVISION = _contract(
    "v12_packager_revision",
    f"microban_teleop_v12_final_deployment_packager_v6_{_TAG}_servo_range",
    f"microban_teleop_v12_final_deployment_packager_v7_{_TAG}_servo_range",
)

# v12 rescue stages (their checkpoints replay HOME-frame hand targets).
V12_FINAL_RESCUE_RECIPE_REVISION = _contract(
    "v12_final_rescue_recipe_revision",
    f"model14900_targeted_final_scenario_replay_to15000_{_TAG}_v1",
    f"model14900_targeted_final_scenario_replay_to15000_{_TAG}_home_levelled_receiver_box_v2",
)
V12_FINAL_RESCUE_MARKER_REVISION = _contract(
    "v12_final_rescue_marker_revision",
    f"recorded_model14900_ordinary10_final_scenarios90_99_updates_{_TAG}_v1",
    f"recorded_model14900_ordinary10_final_scenarios90_99_updates_{_TAG}_v2",
)
V12_FINAL_RESCUE_SAMPLER_REVISION = _contract(
    "v12_final_rescue_sampler_revision",
    f"episode_shared_twist_foot_hand_evaluator_scenario_replay_{_TAG}_v1",
    f"episode_shared_twist_foot_hand_evaluator_scenario_replay_home_levelled_{_TAG}_v2",
)
V12_CORNER_RESCUE_RECIPE_REVISION = _contract(
    "v12_corner_rescue_recipe_revision",
    f"model9900_targeted_bilateral_corner_pair_replay_to10000_{_TAG}_v3",
    f"model9900_targeted_bilateral_corner_pair_replay_to10000_receiver_box_f_{_TAG}_v4",
)
V12_CORNER_RESCUE_MARKER_REVISION = _contract(
    "v12_corner_rescue_marker_revision",
    f"recorded_model9900_uniform5_lf_rb90_lb_rf5_99_updates_{_TAG}_v3",
    f"recorded_model9900_uniform5_lf_rb90_lb_rf5_99_updates_receiver_box_f_{_TAG}_v4",
)
V12_CORNER_RESCUE_SAMPLER_REVISION = _contract(
    "v12_corner_rescue_sampler_revision",
    f"uniform_joint_box5pct_lf_rb90pct_lb_rf5pct_{_TAG}_v2",
    f"uniform_joint_box5pct_lf_rb90pct_lb_rf5pct_receiver_box_f_{_TAG}_v3",
)


def _pose_release_corner_rescue(mix: str, lf_rb: int, lb_rf: int, sampler_suffix: str) -> dict[str, str]:
    box = "" if _UPRIGHT else "receiver_box_f_"
    return {
        "marker_revision": HOME.contract(
            f"v12_pr_corner_rescue_{mix}_marker_revision",
            f"recorded_pose_release_model9900_uniform5_lf_rb{lf_rb}_lb_rf{lb_rf}_99_updates_"
            f"{box}{_TAG}_v1",
        ),
        "sampler_revision": HOME.contract(
            f"v12_pr_corner_rescue_{mix}_sampler_revision",
            f"uniform_joint_box5pct_lf_rb{lf_rb}pct_lb_rf{lb_rf}pct_{box}{sampler_suffix}{_TAG}_v1",
        ),
    }


# Pose-release corner rescue sampler mixes (name -> marker/sampler revision).
V12_POSE_RELEASE_CORNER_RESCUE_MIX_REVISIONS = {
    "lf60": _pose_release_corner_rescue("lf60", 60, 35, ""),
    "lf72": _pose_release_corner_rescue("lf72", 72, 23, "pose_release_"),
    "lf90": _pose_release_corner_rescue("lf90", 90, 5, "pose_release_"),
}

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

# The pinned canonical model_7099 the release-eligible pose-release recipe
# switch may resume (None: no switch; train the pose-release recipe fresh).
V12_POSE_RELEASE_SWITCH_PARENT_SHA256: str | None = HOME.override(  # type: ignore[assignment]
    "v12_pose_release_switch_parent_sha256"
)
# Walking checkpoints without the HOME stamp (saved before it existed).
ACCEPTS_UNSTAMPED_WALK_CHECKPOINTS = bool(HOME.override("accepts_unstamped_walk_checkpoints", False))


def robot_contract_strings() -> dict[str, str]:
    """The HOME-bound identifiers the robot runtime checks (robot config/home_pose.yaml)."""

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
