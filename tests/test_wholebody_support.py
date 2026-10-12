"""The support shift's pieces (retarget/support.py): Microban's zero-moment point from its whole-body dynamics and
the support, the contact hull shrunk by the margin.  The shift itself is tested on a retargeted one-leg stand in
test_wholebody_retarget.py."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

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


def test_the_margin_is_a_length():
    for margin in (-0.001, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            support.SupportSpec(margin_m=margin)
