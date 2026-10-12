"""The support changes' pieces (retarget/support.py): Microban's zero-moment point from its whole-body dynamics,
the support (the contact hull shrunk by the margin), the feet held where they stand and the stages.  The changes
themselves are tested on a retargeted one-leg stand in test_wholebody_retarget.py."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from mjlab_microban.wholebody.retarget import model, support

FPS = 30.0


@pytest.fixture(scope="module")
def robot():
    return mujoco.MjModel.from_xml_path(str(model.gmr_model_xml()))


def _com(m, q):
    d = mujoco.MjData(m)
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    return d.subtree_com[1].copy()


def test_zmp_of_a_still_pose_is_below_the_centre_of_mass(robot):
    home = model.home_qpos(robot)
    z, load = support.zmp(robot, np.tile(home, (12, 1)), FPS, 0.3)
    assert np.abs(z - _com(robot, home)[:2]).max() < 1e-9
    assert np.allclose(load, 1.0)


def test_zmp_of_the_robot_carried_with_an_acceleration(robot):
    # The whole robot (HOME, joints still) accelerates at 2 m/s^2 forward: the floor has to push it forward too, so
    # the zero-moment point is behind the centre of mass by its height x 2 / g.
    home = model.home_qpos(robot)
    t = np.arange(15) / FPS
    qs = np.tile(home, (len(t), 1))
    qs[:, 0] += 0.5 * 2.0 * t**2
    z, load = support.zmp(robot, qs, FPS, 0.3)
    com = np.array([_com(robot, q) for q in qs])
    assert np.abs(z[:, 0] - (com[:, 0] - com[:, 2] * 2.0 / support.GRAVITY)).max() < 1e-6
    assert np.abs(z[:, 1] - com[:, 1]).max() < 1e-9 and np.allclose(load, 1.0)


def _size(region):
    A, h = region
    v = support.region_vertices(A, h)
    return np.ptp(v[:, 0]), np.ptp(v[:, 1])


def test_the_support_is_the_contact_hull_shrunk_by_the_margin():
    sole = np.array([[x, y] for x in (-0.031, 0.031) for y in (-0.02, 0.02)])
    assert _size(support.shrink(*support.halfplanes(sole), 0.010)) == pytest.approx((0.042, 0.020), abs=1e-6)
    assert _size(support.shrink(*support.halfplanes(sole), 0.005)) == pytest.approx((0.052, 0.030), abs=1e-6)
    # Two soles: the hull of both, shrunk.
    both = np.r_[sole + [0.0, 0.0465], sole - [0.0, 0.0465]]
    assert _size(support.shrink(*support.halfplanes(both), 0.010)) == pytest.approx((0.042, 0.113), abs=1e-6)
    # Narrower than twice the margin (a foot on its heel's edge): its middle line, the deepest points.
    strip = np.array([[x, y] for x in (-0.031, 0.031) for y in (-0.0075, 0.0075)])
    length, width = _size(support.shrink(*support.halfplanes(strip), 0.010))
    assert length == pytest.approx(0.047, abs=1e-5) and width < 1e-5


def test_the_margin_and_the_caps_are_finite_and_not_negative():
    for field in ("margin_m", "max_lateral_m", "max_forward_m", "max_tilt_deg", "max_ankle_roll_deg", "max_foot_m"):
        for value in (-0.001, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                support.SupportSpec(**{field: value})
    assert support.SupportSpec(max_tilt_deg=0.0).caps()[2:4].tolist() == [0.0, 0.0]  # 0: the tilt is not used


def test_the_changes_acceleration_moves_the_zmp_by_its_own_differences(robot):
    # The unchanged clip (HOME, still) has zmp's ZMP; changed by carrying the whole robot forward at 2 m/s^2, its ZMP
    # is behind the centre of mass by its height x 2 / g, the changes' accelerations taken from their own 5-frame fit.
    home = model.home_qpos(robot)
    t = np.arange(15) / FPS
    base = np.tile(home, (len(t), 1))
    moved = base.copy()
    moved[:, 0] += 0.5 * 2.0 * t**2
    dyn = support.Dyn(robot, base, FPS, 0.3, 0.17)
    z0, _ = dyn.zmp(base)
    assert np.abs(z0 - support.zmp(robot, base, FPS, 0.3)[0]).max() < 1e-12
    z, load = dyn.zmp(moved)
    com = np.array([_com(robot, q) for q in moved])
    assert np.abs(z[:, 0] - (com[:, 0] - com[:, 2] * 2.0 / support.GRAVITY)).max() < 1e-6 and np.allclose(load, 1.0)


def test_a_standing_foot_is_held_where_it_stands_and_rolls_on_its_toes(robot):
    # The left foot stands over frames 0-14: flat and sliding forward 2 mm a frame with its heading wavering by 1
    # degree over 0-9, then its heel rising to 40 degrees over 10-14 with the ankle going on forward.  Held: flat,
    # the ankle at its median there and the heading at the run's; heel up, the toe's edge stays where the flat foot
    # had it.  After the run the hold blends back without a jump.
    T = 25
    contact = np.zeros((T, len(support.PARTS)), bool)
    contact[:15, support.FEET[0]] = True
    pos = np.zeros((T, 2, 3))
    pos[:, 0, 0] = 0.002 * np.arange(T)
    pitch = np.radians(np.r_[np.zeros(10), np.linspace(10.0, 40.0, 5), np.full(10, 40.0)])
    yaw = np.radians(np.r_[np.where(np.arange(10) % 2, 0.5, -0.5), np.zeros(15)])
    rot = R.from_euler("ZY", np.c_[yaw, pitch])
    quat = np.zeros((T, 2, 4))
    quat[:, 0] = rot.as_quat(scalar_first=True)
    quat[:, 1, 0] = 1.0
    soles = support.sole_points(robot, support.Shapes(robot))
    offset, turn, stand = support.foot_lock(contact, pos, quat, soles)
    held = pos[:, 0, :2] + offset[:, 0]
    assert np.abs(held[:10] - np.median(pos[:10, 0, :2], 0)).max() < 1e-9
    assert np.degrees(np.abs(yaw[:15] + turn[:15, 0])).max() < 0.01
    bottom = soles[0][soles[0][:, 2] <= soles[0][:, 2].min() + 0.001]
    toe = np.r_[bottom[:, 0].max(), bottom.mean(0)[1:]]
    turned = R.from_rotvec(np.c_[np.zeros((5, 2)), yaw[10:15] + turn[10:15, 0]]) * R.from_rotvec(np.c_[np.zeros(5), pitch[10:15], np.zeros(5)])
    toe_at = held[10:15] + turned.apply(toe)[:, :2]
    assert np.abs(toe_at - (held[0] + toe[:2])).max() < 1e-9
    assert np.abs(np.diff(held[14:, 0])).max() < 0.01 and not offset[:, 1].any() and not turn[:, 1].any()
    assert stand[:, 0].tolist() == contact[:, support.FEET[0]].astype(float).tolist() and not stand[:, 1].any()


def test_a_frame_comes_in_at_the_stage_from_which_it_stays_in():
    nan, out, inside = float("nan"), 0.01, 0.0
    dists = [[out, out, inside, out, nan],  # unchanged
             [out, inside, inside, out, nan],  # shift
             [out, out, inside, out, nan],  # tilt
             [inside, inside, inside, out, nan],  # ankle
             [inside, inside, inside, out, nan]]  # foot
    assert support.stage_reached([np.array(d) for d in dists]).tolist() == [3, 3, 0, 5, -1]
