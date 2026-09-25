"""Pure contract tests for the live PICO -> 99-value tracking bridge."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict

from mjlab_microban.live_pico_tracking_bridge import (
    LIVE_TRACKING_ACTOR_SCHEMA,
    LIVE_TRACKING_ACTOR_WIDTH,
    PICO_BODY_JOINT_NAMES,
    LivePicoTrackingReferenceBuilder,
    LiveTrackingBridgeError,
    LiveTrackingReference,
    OnlineMicrobanRetargeter,
    clip_tracking_action_to_soft_limits,
    extract_pico_body_sample,
    patch_tracking_actor_observation,
    tracking_orientation_observation,
)
from mjlab_microban.scripts.live_pico_tracking_sim import (
    LiveFallLatch,
    _assess_live_robot_state,
    _install_live_motion_freeze,
    _left_trigger_released,
)
from mjlab_microban.tasks.microban_policy_export import (
    MICROBAN_TELEOP_ACTION_JOINT_NAMES,
)
from mjlab_microban.tasks.microban_tracking_env_cfg import (
    MICROBAN_BODY_JOINT_SOFT_LIMITS,
)


def _yaw_xyzw(angle: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0))


def _frame(
    timestamp_ns: int,
    *,
    pelvis_x: float = 0.0,
    pelvis_yaw: float = 0.0,
    valid: bool = True,
    fresh: bool = True,
    jumps: tuple[object, ...] = (),
) -> SimpleNamespace:
    joints = []
    for index, name in enumerate(PICO_BODY_JOINT_NAMES):
        position = (pelvis_x if index == 0 else 0.0, float(index) * 0.01, 1.0)
        orientation = _yaw_xyzw(pelvis_yaw) if index == 0 else (0.0, 0.0, 0.0, 1.0)
        joints.append(
            SimpleNamespace(
                index=index,
                name=name,
                pose=SimpleNamespace(position=position, orientation=orientation),
            )
        )
    return SimpleNamespace(
        sampled_at_ns=timestamp_ns,
        body_health=SimpleNamespace(valid=valid, fresh=fresh),
        body_jumps=jumps,
        body=SimpleNamespace(timestamp_ns=timestamp_ns, joints=tuple(joints)),
    )


class _FakeRetargeter:
    def __init__(self) -> None:
        self.home_action_joint_pos = np.asarray(
            [
                sum(MICROBAN_BODY_JOINT_SOFT_LIMITS[name]) / 2.0
                for name in MICROBAN_TELEOP_ACTION_JOINT_NAMES
            ],
            dtype=np.float64,
        )
        self.error = 0.001
        self.extra_delta = 0.0
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def solve(
        self,
        positions_m: np.ndarray,
        *,
        alignment: np.ndarray,
        desired_trunk_matrix: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        del alignment, desired_trunk_matrix
        result = self.home_action_joint_pos.copy()
        result[0] += float(positions_m[0, 0]) + self.extra_delta
        return result, self.error


class LivePicoBodyContractTest(unittest.TestCase):
    def test_exact_actor_schema_is_99_values(self) -> None:
        self.assertEqual(
            LIVE_TRACKING_ACTOR_SCHEMA,
            (
                ("command", 36),
                ("motion_anchor_ori_b", 6),
                ("base_ang_vel", 3),
                ("joint_pos", 18),
                ("joint_vel", 18),
                ("actions", 18),
            ),
        )
        self.assertEqual(LIVE_TRACKING_ACTOR_WIDTH, 99)

    def test_extract_requires_fresh_exact_named_body(self) -> None:
        sample = extract_pico_body_sample(_frame(1))
        self.assertEqual(sample.timestamp_ns, 1)
        self.assertEqual(sample.positions_m.shape, (24, 3))

        stale = _frame(2, fresh=False)
        with self.assertRaisesRegex(LiveTrackingBridgeError, "stale"):
            extract_pico_body_sample(stale)

        wrong = _frame(3)
        wrong.body.joints[5].name = "not_right_knee"
        with self.assertRaisesRegex(LiveTrackingBridgeError, "right_knee"):
            extract_pico_body_sample(wrong)

        jumped = _frame(4, jumps=("pelvis",))
        with self.assertRaisesRegex(LiveTrackingBridgeError, "pose jump"):
            extract_pico_body_sample(jumped)


class LiveReferenceBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.retargeter = _FakeRetargeter()
        self.builder = LivePicoTrackingReferenceBuilder(
            self.retargeter,
            max_body_gap_s=0.1,
            max_ik_error_m=0.02,
            max_joint_speed_rad_s=5.0,
        )
        self.robot_quaternion = (1.0, 0.0, 0.0, 0.0)

    def _update(
        self,
        frame: object,
        *,
        ready: bool,
        enabled: bool,
        robot_ready: bool = True,
    ) -> LiveTrackingReference | None:
        return self.builder.update(
            frame,
            calibration_ready=ready,
            robot_calibration_ready=robot_ready,
            enabled=enabled,
            robot_trunk_quat_wxyz=self.robot_quaternion,
        )

    def test_trigger_release_calibrates_and_hold_generates_reference(self) -> None:
        neutral = self._update(_frame(1_000_000_000), ready=True, enabled=False)
        assert neutral is not None
        self.assertTrue(self.builder.calibrated)
        np.testing.assert_allclose(
            neutral.joint_pos_rad, self.retargeter.home_action_joint_pos
        )
        np.testing.assert_allclose(neutral.joint_vel_rad_s, 0.0)

        live = self._update(
            _frame(1_020_000_000, pelvis_x=0.02, pelvis_yaw=0.1),
            ready=True,
            enabled=True,
        )
        assert live is not None
        self.assertAlmostEqual(
            live.joint_pos_rad[0],
            self.retargeter.home_action_joint_pos[0] + 0.02,
        )
        self.assertAlmostEqual(live.joint_vel_rad_s[0], 1.0)
        orientation = tracking_orientation_observation(
            self.robot_quaternion, live.desired_trunk_quat_wxyz
        )
        np.testing.assert_allclose(
            np.asarray(orientation).reshape(3, 2),
            np.asarray(
                (
                    (math.cos(0.1), -math.sin(0.1)),
                    (math.sin(0.1), math.cos(0.1)),
                    (0.0, 0.0),
                )
            ),
            atol=1.0e-8,
        )

        released = self._update(
            _frame(1_040_000_000, pelvis_x=0.04), ready=True, enabled=False
        )
        assert released is not None
        np.testing.assert_allclose(
            released.joint_pos_rad, self.retargeter.home_action_joint_pos
        )
        np.testing.assert_allclose(released.joint_vel_rad_s, 0.0)

    def test_never_calibrates_while_trigger_is_held(self) -> None:
        with self.assertRaisesRegex(LiveTrackingBridgeError, "released-trigger"):
            self._update(_frame(1_000_000_000), ready=True, enabled=True)
        self.assertFalse(self.builder.calibrated)

    def test_calibration_requires_robot_home_readiness(self) -> None:
        waiting = self._update(
            _frame(1_000_000_000),
            ready=True,
            enabled=False,
            robot_ready=False,
        )
        self.assertIsNone(waiting)
        self.assertFalse(self.builder.calibrated)

        neutral = self._update(
            _frame(1_020_000_000),
            ready=True,
            enabled=False,
            robot_ready=True,
        )
        self.assertIsNotNone(neutral)
        self.assertTrue(self.builder.calibrated)

        invalidated = self._update(
            _frame(1_040_000_000),
            ready=True,
            enabled=False,
            robot_ready=False,
        )
        self.assertIsNone(invalidated)
        self.assertFalse(self.builder.calibrated)

    def test_duplicate_is_idempotent_but_changed_payload_fails_closed(self) -> None:
        frame = _frame(1_000_000_000)
        first = self._update(frame, ready=True, enabled=False)
        self.assertIs(first, self._update(frame, ready=True, enabled=False))

        changed = _frame(1_000_000_000, pelvis_x=0.01)
        with self.assertRaisesRegex(LiveTrackingBridgeError, "without a new timestamp"):
            self._update(changed, ready=True, enabled=False)
        self.assertFalse(self.builder.calibrated)

    def test_timestamp_gap_ik_error_speed_and_soft_limit_all_fail_closed(self) -> None:
        self._update(_frame(1_000_000_000), ready=True, enabled=False)
        with self.assertRaisesRegex(LiveTrackingBridgeError, "timestamp has a gap"):
            self._update(_frame(1_200_000_000), ready=True, enabled=True)
        self.assertFalse(self.builder.calibrated)

        self._update(_frame(2_000_000_000), ready=True, enabled=False)
        self.retargeter.error = 0.03
        with self.assertRaisesRegex(LiveTrackingBridgeError, "IK error"):
            self._update(_frame(2_020_000_000), ready=True, enabled=True)
        self.assertFalse(self.builder.calibrated)

        self.retargeter.error = 0.001
        self._update(_frame(3_000_000_000), ready=True, enabled=False)
        with self.assertRaisesRegex(LiveTrackingBridgeError, "joint speed"):
            self._update(_frame(3_020_000_000, pelvis_x=0.11), ready=True, enabled=True)
        self.assertFalse(self.builder.calibrated)

        self._update(_frame(4_000_000_000), ready=True, enabled=False)
        self.retargeter.extra_delta = 10.0
        with self.assertRaisesRegex(LiveTrackingBridgeError, "soft limit"):
            self._update(_frame(4_020_000_000), ready=True, enabled=True)
        self.assertFalse(self.builder.calibrated)


class TrackingTensorAdapterTest(unittest.TestCase):
    def test_patch_replaces_only_reference_prefix(self) -> None:
        actor = torch.arange(99, dtype=torch.float32).unsqueeze(0)
        observations = TensorDict({"actor": actor}, batch_size=[1])
        reference = LiveTrackingReference(
            tuple(float(value) for value in range(18)),
            tuple(float(-value) for value in range(18)),
            (1.0, 0.0, 0.0, 0.0),
            1,
            0.0,
        )
        patched = patch_tracking_actor_observation(
            observations,
            reference,
            current_trunk_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        self.assertTrue(torch.equal(observations["actor"], actor))
        self.assertTrue(torch.equal(patched["actor"][:, 42:], actor[:, 42:]))
        self.assertTrue(
            torch.equal(
                patched["actor"][0, :36],
                torch.tensor(tuple(range(18)) + tuple(-value for value in range(18))),
            )
        )
        self.assertTrue(
            torch.equal(
                patched["actor"][0, 36:42],
                torch.tensor((1.0, 0.0, 0.0, 1.0, 0.0, 0.0)),
            )
        )

    def test_action_projection_enforces_absolute_soft_limits(self) -> None:
        raw = torch.tensor([[1.0e6] * 18])
        scale = torch.ones(18)
        offset = torch.tensor(
            [
                sum(MICROBAN_BODY_JOINT_SOFT_LIMITS[name]) / 2.0
                for name in MICROBAN_TELEOP_ACTION_JOINT_NAMES
            ]
        )
        lower = torch.tensor(
            [
                MICROBAN_BODY_JOINT_SOFT_LIMITS[name][0]
                for name in MICROBAN_TELEOP_ACTION_JOINT_NAMES
            ]
        )
        upper = torch.tensor(
            [
                MICROBAN_BODY_JOINT_SOFT_LIMITS[name][1]
                for name in MICROBAN_TELEOP_ACTION_JOINT_NAMES
            ]
        )
        projected = clip_tracking_action_to_soft_limits(
            raw, scale=scale, offset=offset, lower=lower, upper=upper
        )
        target = projected * scale + offset
        self.assertTrue(bool((target >= lower).all().item()))
        self.assertTrue(bool((target <= upper).all().item()))
        with self.assertRaisesRegex(LiveTrackingBridgeError, "non-finite"):
            clip_tracking_action_to_soft_limits(
                torch.full((1, 18), float("nan")),
                scale=scale,
                offset=offset,
                lower=lower,
                upper=upper,
            )

    def test_real_retargeter_home_is_within_policy_limits(self) -> None:
        retargeter = OnlineMicrobanRetargeter()
        for value, name in zip(
            retargeter.home_action_joint_pos,
            MICROBAN_TELEOP_ACTION_JOINT_NAMES,
            strict=True,
        ):
            lower, upper = MICROBAN_BODY_JOINT_SOFT_LIMITS[name]
            self.assertGreaterEqual(value, lower)
            self.assertLessEqual(value, upper)

    def test_real_retargeter_does_not_drift_on_identical_pose(self) -> None:
        positions = np.zeros((len(PICO_BODY_JOINT_NAMES), 3), dtype=np.float64)
        index = {name: i for i, name in enumerate(PICO_BODY_JOINT_NAMES)}
        for side, lateral in (("left", 0.1), ("right", -0.1)):
            positions[index[f"{side}_hip"]] = (0.0, lateral, 0.0)
            positions[index[f"{side}_knee"]] = (0.0, lateral, -0.4)
            positions[index[f"{side}_foot"]] = (0.05, lateral, -0.8)
            positions[index[f"{side}_shoulder"]] = (0.0, 2.0 * lateral, 0.5)
            positions[index[f"{side}_elbow"]] = (0.0, 5.0 * lateral, 0.4)
            positions[index[f"{side}_hand"]] = (0.0, 8.0 * lateral, 0.3)
        retargeter = OnlineMicrobanRetargeter()
        first, first_error = retargeter.solve(
            positions, alignment=np.eye(3), desired_trunk_matrix=np.eye(3)
        )
        second, second_error = retargeter.solve(
            positions, alignment=np.eye(3), desired_trunk_matrix=np.eye(3)
        )
        np.testing.assert_allclose(second, first, atol=1.0e-10, rtol=0.0)
        self.assertAlmostEqual(second_error, first_error, places=12)

    def test_real_retargeter_reports_final_maximum_endpoint_error(self) -> None:
        positions = np.zeros((len(PICO_BODY_JOINT_NAMES), 3), dtype=np.float64)
        index = {name: i for i, name in enumerate(PICO_BODY_JOINT_NAMES)}
        for side, lateral in (("left", 0.1), ("right", -0.1)):
            positions[index[f"{side}_hip"]] = (0.0, lateral, 0.0)
            positions[index[f"{side}_knee"]] = (0.1, lateral, -0.3)
            positions[index[f"{side}_foot"]] = (0.2, lateral, -0.6)
            positions[index[f"{side}_shoulder"]] = (0.0, 2.0 * lateral, 0.5)
            positions[index[f"{side}_elbow"]] = (0.1, 4.0 * lateral, 0.4)
            positions[index[f"{side}_hand"]] = (0.2, 6.0 * lateral, 0.3)

        # A one-iteration solve exercises the former stale pre-step residual bug.
        retargeter = OnlineMicrobanRetargeter(max_iterations=1)
        _solution, maximum_error = retargeter.solve(
            positions, alignment=np.eye(3), desired_trunk_matrix=np.eye(3)
        )
        endpoint_errors = []
        for kind, body_id, target, _weight in retargeter._targets(positions, np.eye(3)):
            current, _jacobian = retargeter._position_jacobian(kind, body_id)
            endpoint_errors.append(float(np.linalg.norm(target - current)))
        self.assertAlmostEqual(maximum_error, max(endpoint_errors), places=12)
        self.assertTrue(math.isfinite(retargeter.last_weighted_rms_error_m))
        self.assertLessEqual(retargeter.last_weighted_rms_error_m, maximum_error)


class LiveRobotSafetyTest(unittest.TestCase):
    @staticmethod
    def _assessment(
        *,
        height: float = 0.168,
        quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        root_speed: float = 0.0,
        body_error: float = 0.0,
    ) -> object:
        home = torch.zeros((1, 18), dtype=torch.float32)
        body = home.clone()
        body[0, 0] = body_error
        return _assess_live_robot_state(
            root_pos_w=torch.tensor(((0.0, 0.0, height),)),
            root_quat_wxyz=torch.tensor((quaternion,)),
            root_lin_vel=torch.tensor(((root_speed, 0.0, 0.0),)),
            root_ang_vel=torch.zeros((1, 3)),
            body_joint_pos=body,
            body_joint_vel=torch.zeros((1, 18)),
            home_body_joint_pos=home,
            environment_origin_z=0.0,
        )

    def test_health_and_calibration_have_distinct_thresholds(self) -> None:
        healthy = self._assessment()
        self.assertIsNone(healthy.fall_reason)
        self.assertTrue(healthy.calibration_ready)

        moving = self._assessment(root_speed=0.1)
        self.assertIsNone(moving.fall_reason)
        self.assertIn("moving", moving.calibration_block_reason or "")

        away_from_home = self._assessment(body_error=0.2)
        self.assertIsNone(away_from_home.fall_reason)
        self.assertIn("HOME", away_from_home.calibration_block_reason or "")

    def test_height_tilt_and_nonfinite_state_latch_as_falls(self) -> None:
        self.assertIn("height", self._assessment(height=0.05).fall_reason or "")
        sideways = (
            math.sqrt(0.5),
            math.sqrt(0.5),
            0.0,
            0.0,
        )
        self.assertIn(
            "up-vector",
            self._assessment(quaternion=sideways).fall_reason or "",
        )
        self.assertIn(
            "non-finite", self._assessment(height=float("nan")).fall_reason or ""
        )

    def test_fall_latch_requires_safe_released_trigger_frame(self) -> None:
        latch = LiveFallLatch()
        self.assertEqual(
            latch.update(fall_reason="fallen", left_trigger_released=False),
            (True, True, False),
        )
        self.assertEqual(
            latch.update(fall_reason=None, left_trigger_released=False),
            (True, False, False),
        )
        self.assertEqual(
            latch.update(fall_reason="still fallen", left_trigger_released=True),
            (True, False, False),
        )
        self.assertEqual(
            latch.update(fall_reason=None, left_trigger_released=True),
            (False, False, True),
        )

    def test_trigger_release_is_read_from_analog_trigger_not_policy_state(self) -> None:
        released = SimpleNamespace(left_controller=SimpleNamespace(trigger=0.0))
        held = SimpleNamespace(left_controller=SimpleNamespace(trigger=0.8))
        self.assertTrue(_left_trigger_released(released, threshold=0.15))
        self.assertFalse(_left_trigger_released(held, threshold=0.15))
        self.assertFalse(_left_trigger_released(SimpleNamespace(), threshold=0.15))


class LiveMotionFreezeTest(unittest.TestCase):
    def test_freeze_holds_phase_and_resets_all_joints_to_home(self) -> None:
        class FakeRobot:
            def __init__(self) -> None:
                self.data = SimpleNamespace(
                    default_root_state=torch.tensor(
                        (
                            (
                                0.0,
                                0.0,
                                0.168,
                                1.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                            ),
                        )
                    ),
                    default_joint_pos=torch.arange(21, dtype=torch.float32).unsqueeze(
                        0
                    ),
                    default_joint_vel=torch.zeros((1, 21)),
                )
                self.root_state = torch.full((1, 13), float("nan"))
                self.joint_pos = torch.full((1, 21), float("nan"))
                self.joint_vel = torch.full((1, 21), float("nan"))
                self.reset_count = 0

            def write_root_state_to_sim(
                self, value: torch.Tensor, *, env_ids: torch.Tensor
            ) -> None:
                self.root_state[env_ids] = value

            def write_joint_state_to_sim(
                self,
                position: torch.Tensor,
                velocity: torch.Tensor,
                *,
                env_ids: torch.Tensor,
            ) -> None:
                self.joint_pos[env_ids] = position
                self.joint_vel[env_ids] = velocity

            def reset(self, *, env_ids: torch.Tensor) -> None:
                del env_ids
                self.reset_count += 1

        class FakeCommand:
            def __init__(self, robot: FakeRobot) -> None:
                self.robot = robot
                self.time_steps = torch.tensor((267,), dtype=torch.long)
                self.metrics_count = 0
                self.relative_count = 0

            def _update_metrics(self) -> None:
                self.metrics_count += 1

            def update_relative_body_poses(self) -> None:
                self.relative_count += 1

        robot = FakeRobot()
        command = FakeCommand(robot)
        scene = SimpleNamespace(
            env_origins=torch.tensor(((2.0, 3.0, 4.0),)),
        )
        env = SimpleNamespace(
            num_envs=1,
            device="cpu",
            scene=scene,
            command_manager=SimpleNamespace(get_term=lambda name: command),
        )
        command._env = env

        frozen = _install_live_motion_freeze(env)
        self.assertIs(frozen, command)
        self.assertEqual(int(command.time_steps.item()), 0)
        torch.testing.assert_close(
            robot.root_state[0, :3], torch.tensor((2.0, 3.0, 4.168))
        )
        torch.testing.assert_close(robot.joint_pos, robot.data.default_joint_pos)
        torch.testing.assert_close(robot.joint_vel, robot.data.default_joint_vel)

        for _ in range(2 * 268 + 5):
            command.compute(1.0e9)
        self.assertEqual(int(command.time_steps.item()), 0)
        self.assertFalse(command.apply_gui_reset(torch.tensor((0,))))

        robot.joint_pos.fill_(99.0)
        command._resample_command(torch.tensor((0,)))
        torch.testing.assert_close(robot.joint_pos, robot.data.default_joint_pos)


if __name__ == "__main__":
    unittest.main()
