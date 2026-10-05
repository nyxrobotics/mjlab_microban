"""Balance check of config/home_pose.yaml for scripts/retrain_all_for_home.py.

usage: uv run --locked python scripts/home_pipeline/home_check.py [--yaml PATH]

Prints one JSON object and never modifies the YAML.  Two definitions of the
sole contact area are reported:

* ``loader``: the one of ``home_pose.analyze_pose`` (and of
  ``config/balance_home_pose.py``): the sole box corners lowest along each
  foot's own normal.  It assumes a sole that is level in pitch AND roll.
* ``ground``: the corners within 1e-4 m of the lowest world height, i.e. the
  corners that really touch a flat floor when the robot stands at HOME (the
  definition of the earlier forward-lean derivation).  A rolled sole touches
  only along an edge, so this range can be much shorter.

``status`` is ``refuse`` when the COM lies outside the fore-aft sole range
under either definition, ``warn`` when |COM - sole centre| exceeds 0.5 mm
under either definition or fewer corners touch the floor than the loader
counts (a sole rolled beyond the 1e-4 m band), else ``ok``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mjlab_microban.robot import home_pose  # noqa: E402

GROUND_CONTACT_BAND_M = 1.0e-4
WARN_OFFSET_M = 0.5e-3


def _ground_contact(hp: home_pose.HomePose) -> dict[str, object]:
    """Contact range from the world height of every sole corner."""

    model = home_pose._pose_model(str(home_pose.MICROBAN_XML.resolve()))
    model.set_pose(dict(hp.joint_pos_rad), hp.trunk_pitch_rad)
    corners = np.concatenate([model.sole_corners("left"), model.sole_corners("right")])
    z = corners[:, 2] - corners[:, 2].min()
    low = corners[z < GROUND_CONTACT_BAND_M]
    com_x = float(model.data.subtree_com[model.trunk_id][0])
    x_min, x_max = float(low[:, 0].min()), float(low[:, 0].max())
    rolls = []
    for side in ("left", "right"):
        normal = model.sole_normal(side)
        rolls.append(math.degrees(math.atan2(-normal[1], normal[2])))
    return {
        "contact_corner_count": int(len(low)),
        "sole_x_range_m": [x_min, x_max],
        "com_offset_x_m": com_x - 0.5 * (x_min + x_max),
        "heel_margin_m": com_x - x_min,
        "toe_margin_m": x_max - com_x,
        "sole_roll_deg": rolls,
    }


def check(path: Path) -> dict[str, object]:
    try:
        hp = home_pose.load_home_pose(path)
    except Exception as error:  # noqa: BLE001 - report every loader refusal
        return {
            "status": "refuse",
            "reasons": [home_pose.describe_home_yaml_error(error, path)],
            "warnings": [],
        }
    a = hp.analysis
    loader = {
        "contact_corner_count": a.sole_contact_corner_count,
        "sole_x_range_m": [a.sole_x_min, a.sole_x_max],
        "com_offset_x_m": a.com_offset_x,
        "heel_margin_m": a.heel_margin_m,
        "toe_margin_m": a.toe_margin_m,
        "sole_pitch_deg": [math.degrees(v) for v in a.sole_pitch_rad],
    }
    ground = _ground_contact(hp)
    reasons: list[str] = []
    warnings: list[str] = []
    for name, d in (("loader", loader), ("ground", ground)):
        if d["heel_margin_m"] < 0.0 or d["toe_margin_m"] < 0.0:
            reasons.append(
                f"COM is outside the sole ({name} contact definition): heel margin "
                f"{d['heel_margin_m'] * 1e3:+.2f} mm, toe margin {d['toe_margin_m'] * 1e3:+.2f} mm"
            )
        elif abs(d["com_offset_x_m"]) > WARN_OFFSET_M:
            warnings.append(
                f"COM is {d['com_offset_x_m'] * 1e3:+.3f} mm from the fore-aft sole centre "
                f"({name} contact definition; > 0.5 mm): consider "
                "uv run python config/balance_home_pose.py --write"
            )
    if ground["contact_corner_count"] < loader["contact_corner_count"]:
        warnings.append(
            "the soles are not level in roll (sole roll "
            + "/".join(f"{v:+.3f}" for v in ground["sole_roll_deg"])
            + f" deg): only {ground['contact_corner_count']} of "
            f"{loader['contact_corner_count']} sole corners touch the floor"
        )
    return {
        "status": "refuse" if reasons else ("warn" if warnings else "ok"),
        "reasons": reasons,
        "warnings": warnings,
        "path": str(hp.path),
        "name": hp.name,
        "label": hp.label,
        "tag": hp.tag,
        "joint_hash": hp.joint_hash,
        "trunk_pitch_deg": hp.trunk_pitch_deg,
        "joint_pos_deg": dict(hp.joint_pos_deg),
        "root_pos_m": list(hp.root_pos),
        "head_standing_height_m": hp.head_standing_height_m,
        "feet_lateral_m": hp.feet_lateral_m,
        "loader_contact": loader,
        "ground_contact": ground,
        # Clocks whose stage gates a pose-release PICO package must record
        # (home_contracts.V12_HAND_RMS_40MM_BOUNDARY_PROFILES; none at the
        # centered HOME).
        "v12_required_boundary_gate_clocks": (
            [10_000, 10_100] if bool(hp.override("v12_hand_rms_40mm_boundary_profiles", True)) else []
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yaml", type=Path, default=home_pose.HOME_POSE_YAML)
    args = parser.parse_args()
    result = check(args.yaml)
    print(json.dumps(result, indent=1, sort_keys=True))
    return 1 if result["status"] == "refuse" else 0


if __name__ == "__main__":
    sys.exit(main())
