# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Microban get-up (fall recovery) environment.

Separate task and policy from microban_velocity_env_cfg: getting up from an arbitrary
fallen pose (front/back/side, or any limb configuration from a cold power-on) has a
different contact/trajectory structure than steady-state walking, so training it
alongside locomotion risks interference. See microban_teleop/docs/getup_research.md
for the survey this follows (HumanUP, HoST, FRASA, ANYmal) and the switch heuristic
(height/orientation threshold, handled outside this env, in the real control loop).

All 21 joints are actuated here (including neck/head/arms) — getting up needs every
available DOF, and the "keep the neck predictable" concern that excludes it from the
walking policy doesn't apply during a recovery maneuver.
"""

import numpy as np
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.rewards import joint_torques_l2
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microban.robot.microban_constants import MICROBAN_ROBOT_CFG
from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg

from mjlab.envs.mdp import dr
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from mjlab_microban.tasks.mdp import (
    step_based_staged_curriculum,
    reward_based_staged_curriculum,
    head_height_reward,
    home_pose_reward,
    hold_airborne,
    standing_bonus,
    on_feet_reward,
    standing_torque_penalty,
    home_stillness_reward,
    extreme_joint_velocity,
    reset_near_home_fraction,
    hands_released_reward,
)

STANDING_HEIGHT = 0.168  # trunk height when standing (HOME_FRAME.pos z, microban_constants.py)

SCENE_CFG = SceneCfg(
    terrain=TerrainEntityCfg(
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=0,
    ),
    num_envs=1,
    extent=2.0,
    entities={"robot": MICROBAN_ROBOT_CFG},
)

VIEWER_CONFIG = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    entity_name="robot",
    body_name="trunk",
    distance=3.0,
    elevation=-15.0,
    azimuth=90.0,
)

SIM_CFG = SimulationCfg(
    mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        ccd_iterations=100,
    ),
    # Get-up's extreme fallen/tangled poses generate far more simultaneous contacts
    # than steady-state walking; the auto-sized default overflowed ("nefc overflow"
    # warnings, ~700-800 needed) within the first few dozen training iterations.
    # nconmax is left at its auto-sized default: it wasn't the one overflowing, and
    # setting it explicitly (even at a seemingly modest value) blew GPU memory — it's
    # apparently a much larger per-env allocation than njmax at the same number.
    njmax=1200,
    # Same root cause, different counter: self_collision's contact sensor (trunk
    # subtree vs. itself) overflowed its match-count buffer (default 64) once njmax
    # stopped silently capping how many contacts got that far — observed climbing
    # past 90 within seconds of training. mjlab's own G1/Go1 configs already bump this
    # to 500 for similar reasons.
    contact_sensor_maxmatch=500,
)


def make_microban_getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_velocity_env_cfg()

    cfg.viewer = deepcopy(VIEWER_CONFIG)
    cfg.sim = deepcopy(SIM_CFG)
    cfg.scene = deepcopy(SCENE_CFG)

    #---------------------------- Sensors ---------------------------
    self_collision_sensor_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    # Needed so the policy can tell "both feet off the ground" from "on the ground" —
    # the airborne-compliance reward (see below) is gated on this, and the actor
    # observation of it (also below) is what lets that distinction survive to
    # inference, where there's no ground-truth "being held" flag to read instead.
    feet_ground_sensor_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="subtree", pattern=r"^(foot|foot_2)$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    # Detects hand/forearm-ground contact ("radius"/"radius_2" are the forearm
    # bodies the hand sites sit on — see robot.xml) so hands_released_reward (below)
    # can tell "propped up on hands" from "hands free" — added after watching a live
    # training-in-progress rollout get stuck propped up on its hands at ~half
    # height, never releasing them to stand on its feet alone.
    hands_ground_sensor_cfg = ContactSensorCfg(
        name="hands_ground_contact",
        primary=ContactMatch(mode="body", pattern=r"^(radius|radius_2)$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
    )
    cfg.scene.sensors = (self_collision_sensor_cfg, feet_ground_sensor_cfg, hands_ground_sensor_cfg)

    #---------------------------- Terrain ---------------------------
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    #---------------------------- Actions ---------------------------
    # Excludes head/neck_roll/neck_pitch, matching the walking policy's own exclusion
    # (microban_velocity_env_cfg.py) — get-up shouldn't move the neck either. Their
    # default pose is 0.0 for all three, so they don't need a hold_at_default_pose
    # event: EntityData.joint_pos_target resetting excluded joints to 0.0 every episode
    # (the bug the walking policy's own comment documents) is a no-op when the default
    # already IS 0.0, unlike e.g. the elbows/shoulders whose defaults are far from zero.
    dofs_filter = r".*(?<!head)(?<!neck_roll)(?<!neck_pitch)$"
    cfg.actions["joint_pos"].actuator_names = (dofs_filter,)
    cfg.actions["joint_pos"].scale = 1.0
    # Unclipped, early PPO exploration (init_std=1.0, no offset limit) occasionally
    # commands multi-radian targets on some joint, which — combined with 21 joints
    # moving at once from an already-extreme randomized fallen pose — produced a
    # genuine MuJoCo solver blowup (NaN qpos/qvel) reproduced at 4096 envs within ~70
    # steps of real training, well before hold_airborne's own >=2s cooldown could even
    # fire once. A flat +-90 deg clip is generic (not per-joint-tuned) but removes the
    # wild-target tail while still leaving room to discover a get-up motion; matches
    # HoST's "start with a restrictive action bound" curriculum idea, as a static
    # bound rather than a scheduled one for now.
    cfg.actions["joint_pos"].clip = {r".*": (-1.57, 1.57)}

    #---------------------------- Commands ---------------------------
    # No velocity command: this task is "get up", not "walk somewhere".
    del cfg.commands["twist"]

    #---------------------------- Observations ----------------------
    del cfg.observations["actor"].terms["command"]
    del cfg.observations["actor"].terms["base_lin_vel"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["command"]
    del cfg.observations["critic"].terms["height_scan"]
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["critic"].terms["foot_air_time"]
    del cfg.observations["critic"].terms["foot_contact"]
    del cfg.observations["critic"].terms["foot_contact_forces"]
    # projected_gravity/base_ang_vel keep the base config's defaults. joint_pos/
    # joint_vel/actions are filtered to dofs_filter (matching the action space, same
    # as the walking policy) rather than left at all 21 joints — "actions" naturally
    # shrinks to 18 with the action space above, so filtering joint_pos/joint_vel too
    # keeps everything consistent rather than mixing 21- and 18-wide state.
    cfg.observations["actor"].terms["joint_pos"] = ObservationTermCfg(
        func=velocity_mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(dofs_filter,))},
    )
    cfg.observations["actor"].terms["joint_vel"] = ObservationTermCfg(
        func=velocity_mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(dofs_filter,))},
    )
    cfg.observations["critic"].terms["joint_pos"] = deepcopy(cfg.observations["actor"].terms["joint_pos"])
    cfg.observations["critic"].terms["joint_vel"] = deepcopy(cfg.observations["actor"].terms["joint_vel"])

    # foot_contact is NOT added to the actor observation (unlike an earlier version
    # of this config): the real robot has no foot-contact/pressure sensor hardware
    # or read path (src/observer.py's RobotState has no such field), so a policy
    # trained expecting it would need a fabricated/proxy value fed at inference —
    # a real train/deploy mismatch, discovered when preparing this checkpoint for
    # real-robot deployment. Dropping it keeps the deployed policy's input exactly
    # what the hardware can actually measure (IMU + motor encoders only, same as
    # the walking policy, which also has no foot_contact input).

    #---------------------------- Rewards ---------------------------
    del cfg.rewards["track_linear_velocity"]
    del cfg.rewards["track_angular_velocity"]
    del cfg.rewards["pose"]  # tied to the (now-removed) twist command's standing/walking/running regimes
    del cfg.rewards["air_time"]
    del cfg.rewards["foot_clearance"]
    del cfg.rewards["foot_swing_height"]
    del cfg.rewards["foot_slip"]
    del cfg.rewards["soft_landing"]

    # Dropped (was on by default from the base config): it ran ungated for the whole
    # episode, including during the actual recovery motion — where rolling/tumbling
    # through non-upright orientations is exactly what's needed, so an always-on "be
    # upright" pressure fights it. The orientation check that actually matters (once
    # standing) already lives inside standing_bonus, gated to only apply there.
    del cfg.rewards["upright"]

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk",)
    cfg.rewards["body_ang_vel"].weight = -0.05

    cfg.rewards["angular_momentum"].weight = -0.01

    cfg.rewards["action_rate_l2"].weight = -0.02  # lighter than walking: getting up needs large motions

    # Head height instead of trunk height: trunk-height-alone can't tell "right-side
    # up" from "upside down" (both have the trunk high), and once head height is also
    # rewarded, a separate trunk-height term is redundant — head height already only
    # goes up when the trunk is BOTH elevated and correctly oriented (an inverted pose
    # has the head low even with the trunk high), and it's just as orientation-blind
    # while still flat as the trunk version was (same world-Z-height shape). Keeping
    # both would let a high-but-inverted trunk still collect partial credit from the
    # trunk term, fighting the very thing head height is meant to fix.
    # HEAD_STANDING_HEIGHT: the TRUE standing-pose head height, computed directly via
    # forward kinematics from HOME_FRAME (mujoco.mj_forward on the home keyframe,
    # "head" body world Z) — 0.294m.
    #
    # A previous version of this constant (0.260) was deliberately LOWERED from this
    # same theoretical value to match what an earlier, imperfect policy happened to
    # converge to in practice ("no point rewarding height the motion doesn't
    # actually reach") — but that reasoning is circular: capping the target at
    # whatever height the policy already reaches guarantees the policy never has to
    # reach any higher. Confirmed as a real cause (not just theoretically) by
    # watching a live rollout of a training-in-progress checkpoint: it settles into
    # a stable SEATED posture — which a 0.260m head-height bar (and the
    # upright-error metric, which a seated trunk can also satisfy) doesn't
    # distinguish from standing — rather than fully extending its legs. Restored to
    # the true kinematic value so seated no longer satisfies the target.
    HEAD_STANDING_HEIGHT = 0.294
    # Despite the name, _head_height (mdp.py) no longer reads the actual "head" body
    # through this — it only uses .name to resolve the robot entity, then computes a
    # virtual point above the TRUNK's own center of mass/orientation (fixed offset,
    # see _head_height's docstring for why: this task actuates the neck, so tracking
    # the literal head body would let the policy game "head height" via neck
    # articulation alone). Kept as body_names=("head",) only for readability at
    # other call sites, not because anything resolves those body_ids anymore.
    HEAD_ASSET_CFG = SceneEntityCfg("robot", body_names=("head",))
    # 0.95 was tried (the user asked to raise it from 0.85) and MEASURED to create a
    # genuine plateau: head height stalled at ~0.234m (80% of target) from iteration
    # 5000 through 10000 with zero further improvement, and standing_bonus/
    # standing_pose (both gated on this same threshold) stayed near-zero the whole
    # time — the policy could get most of the way up but essentially never
    # sustained crossing the stricter bar, so those two terms never got enough
    # on-policy signal to do their job. Reverted to 0.85 (the value the best run so
    # far — height 0.249m/upright 0.288 by iteration 1750 — used).
    HEAD_STANDING_THRESHOLD = HEAD_STANDING_HEIGHT * 0.85
    # Weight raised 6.0 -> 8.0: standing_bonus's separate trunk-upright check was
    # just dropped (see standing_bonus's docstring) since the TRUE standing head
    # height target makes it redundant — a seated/kneeling/inverted pose physically
    # cannot reach it. That budget is folded in here instead, since head height
    # (now against the correct, un-lowered target) is doing that job on its own.
    cfg.rewards["head_height"] = RewardTermCfg(
        func=head_height_reward,
        weight=36.0,
        params={
            "target_height": HEAD_STANDING_HEIGHT,
            "asset_cfg": HEAD_ASSET_CFG,
            "sensor_name": feet_ground_sensor_cfg.name,
            # power=1.0: no extra steepness bonus, just the plain linear-rise/
            # linear-fall peaked shape (shape = clamp(1-|frac-1|,0,1)). Tried
            # power=3 and power=1.4 first, both add a steeper-near-target bonus on
            # top of the linear shape, but measured directly that even a partial
            # power bonus isn't worth its added complexity/tuning risk here — the
            # weight increase below (8.0 -> 12.0) gives the same "push harder
            # toward the true target" effect the power bonus was meant to provide,
            # more simply and without the early-gradient risk power>1 carries.
            "power": 1.0,
        },
    )

    # Second, separate head-height term at power=2.0 — added on top of the linear
    # one above rather than raising that one's own power in place, so the linear
    # term keeps providing its already-proven, never-vanishing baseline gradient
    # (see head_height's own comment/head_height_reward's docstring for that
    # history) while this one ADDS an accelerating, "closer-to-target pays
    # disproportionately more" bonus: a flat linear reward pays the same marginal
    # amount for the last 10% of the height gap as the first 10%, which measured
    # out as a real plateau (height settling ~80% of target, see standing_bonus's
    # own comment) — nothing in a purely linear reward specifically discourages
    # settling for "pretty tall" over finishing the climb to true standing height.
    # Own weight, independently tunable from the linear term's, rather than
    # folding into it via a shared blend ratio.
    cfg.rewards["head_height_sq"] = RewardTermCfg(
        func=head_height_reward,
        weight=20.0,
        params={
            "target_height": HEAD_STANDING_HEIGHT,
            "asset_cfg": HEAD_ASSET_CFG,
            "sensor_name": feet_ground_sensor_cfg.name,
            "power": 2.0,
        },
    )

    # head_drop_penalty (explicit penalty for downward head velocity, to discourage
    # raise-then-drop bobbing) is dropped — head_height's own weight was raised
    # instead, which turned out to matter more than the drop penalty did.

    # HoST's "post-task" mechanism (arXiv:2502.08378): a reward that only pays out once
    # standing is (nearly) reached, so completing it earlier and holding it accumulates
    # strictly more over the episode. height+upright alone measurably plateau around
    # 80% of target height without ever finishing the motion (see handling_research.md)
    # — this term exists specifically to close that gap, not to duplicate height/upright.
    # "is standing" is gated on head height (HEAD_ASSET_CFG) everywhere below;
    # orientation itself still reads the trunk body specifically (asset_cfg).
    cfg.rewards["standing_bonus"] = RewardTermCfg(
        func=standing_bonus,
        weight=5.0,
        params={
            # Own threshold (0.9), higher than standing_pose's (0.8, below) — the
            # payout keeps scaling up to the TRUE target height above that anyway
            # (fixes a measured plateau where height stalled almost exactly at
            # whatever threshold this gate used, since a flat 1.0 the instant it's
            # crossed gave zero incentive to keep pushing to target_height once
            # banked), so this gate itself can afford to sit later than
            # standing_pose's without reintroducing a similar hard stopping point.
            "height_threshold": 0.9 * HEAD_STANDING_HEIGHT,
            "target_height": HEAD_STANDING_HEIGHT,
            "head_asset_cfg": HEAD_ASSET_CFG,
        },
    )

    # Distinguishes genuinely standing-on-feet from other stable-but-not-standing
    # configurations (sitting/kneeling) that height+upright alone can't tell apart.
    cfg.rewards["on_feet"] = RewardTermCfg(
        func=on_feet_reward,
        weight=2.0,
        params={
            "sensor_name": feet_ground_sensor_cfg.name,
            "target_height": HEAD_STANDING_HEIGHT,
            "head_asset_cfg": HEAD_ASSET_CFG,
        },
    )

    # Rewards letting go of the ground with the hands once already about halfway up
    # (0.5 * HEAD_STANDING_HEIGHT — deliberately much earlier than
    # HEAD_STANDING_THRESHOLD, which gates on_feet/standing_bonus/foot_flat once
    # nearly done). Added after watching a live rollout of a training-in-progress
    # checkpoint prop itself up on its hands/forearms and get stuck there — nothing
    # else in this reward stack ever penalizes staying propped up once the trunk is
    # clearly already elevated enough not to need it, so "prop up and stop" was a
    # stable resting point under the reward stack as it stood.
    cfg.rewards["hands_released"] = RewardTermCfg(
        func=hands_released_reward,
        weight=1.5,
        params={
            "height_threshold": 0.5 * HEAD_STANDING_HEIGHT,
            "sensor_name": hands_ground_sensor_cfg.name,
            "head_asset_cfg": HEAD_ASSET_CFG,
        },
    )

    # foot_flat_reward (explicit "sole level with the ground" check, on top of
    # on_feet's contact-only check) dropped — with standing_pose now weighted
    # heavily (20.0) and gated on genuinely being up, home-pose-matching alone
    # already implies flat feet (the home keyframe's ankle angles ARE flat), so a
    # separate term for it is redundant complexity rather than a real additional
    # constraint (same reasoning as dropping standing_bonus's separate upright
    # check once head height alone already implied it).

    # Hand-support shaping (bracing/releasing a hand/forearm during recovery) is
    # dropped for now: after several iterations (flat reward -> velocity-scaled ->
    # thresholded +/0/-1 -> thresholded +/0) it was still producing worse overall
    # behavior than leaving it out, more complexity than it was earning back. Head
    # height (+ head_drop) is the priority signal; hands can come back later if the
    # legs-and-torso motion alone isn't enough.

    # Once standing, push toward whichever stance costs the least holding torque —
    # observed behavior without this was a wide leg-split with heavily tilted ankles
    # (satisfies height/upright/on-feet, but is not how this robot would naturally
    # balance). A torque-cost signal is a more principled way to discourage that than
    # penalizing deviation from a specific hand-picked default pose.
    cfg.rewards["standing_torque"] = RewardTermCfg(
        func=standing_torque_penalty,
        weight=-0.5,
        params={
            "height_threshold": HEAD_STANDING_THRESHOLD,
            "head_asset_cfg": HEAD_ASSET_CFG,
            "asset_cfg": SceneEntityCfg("robot", actuator_names=(dofs_filter,)),
        },
    )

    # Trembling/oscillation suppression, in terms of the commanded joint TARGET
    # (asset.data.joint_pos_target) — never measured joint_pos/joint_vel, and NOT
    # env.action_manager.action either (that field turned out to be the RAW,
    # pre-clip network output, empirically unbounded — see
    # home_stillness_reward's own docstring for the full history of earlier
    # versions of this idea, including that dead end). Gated on height too (not
    # just target-near-home): only reward holding still once actually standing,
    # matching every other "are we done" reward here.
    cfg.rewards["home_stillness"] = RewardTermCfg(
        func=home_stillness_reward,
        # 5.0 -> 50.0: raw value (pose_closeness * target_stillness) currently
        # tiny (~0.003), safe to scale up since weight is a linear multiplier on
        # an already-smooth piecewise-linear function (no saturation/underflow
        # risk the way the earlier std/threshold miscalibrations had). Not going
        # all the way to 100x-500 in one jump: max possible per-step contribution
        # at weight=50 (pose_closeness=target_stillness=1) is 50, already on par
        # with standing_pose's own max (30) — plenty to compete without instantly
        # dwarfing every other term and destabilizing the value function via a
        # too-rare, too-large reward spike. 50.0 is the FINAL weight — starts at
        # 0.0 (fully off) here, enabled by pose_curriculum below only once
        # standing_pose has already converged reasonably: asking the policy to
        # hold a commanded target still near home before it can even reliably
        # match that pose yet would be a premature, likely counterproductive
        # constraint.
        weight=0.0,
        params={
            "height_threshold": HEAD_STANDING_THRESHOLD,
            # Both measured directly (a debug rollout filtered to standing-only
            # steps, deterministic policy — see the checked-in
            # scratchpad/debug_stillness.py): target_pose_error currently sits at
            # median 1.15rad / p90 1.37rad / max 1.65rad even while standing,
            # and target_rate at median 37.7 / p90 57.4 / max 92.2 rad/sec. The
            # first attempt (1.5, 30) left target_rate's factor saturated at
            # exactly 0 for essentially every standing step (median 37.7 > worst
            # 30), zeroing the whole multiplicative reward regardless of
            # target_pose_worst being reasonably calibrated already. Both raised
            # above today's observed max so there's real (if currently small)
            # headroom on both factors instead of one dominating via saturation.
            "target_pose_worst": 1.7,
            "target_rate_worst": 60.0,
            "head_asset_cfg": HEAD_ASSET_CFG,
            "asset_cfg": SceneEntityCfg("robot", joint_names=(dofs_filter,)),
        },
    )

    # Pull toward the natural default (home) pose — redesigned after surveying the
    # fall-recovery RL literature (HoST arXiv:2502.08378, FRASA arXiv:2410.08655 —
    # Rhoban's own fall-recovery policy for a similarly small biped, HumanUP
    # arXiv:2502.12152) for the "gets up but settles into the wrong posture"
    # failure mode.
    #
    # An earlier version of this fix (broadened gate to ~40% of standing height,
    # weight ramped 3.0->7.0 and std tightened 0.2-0.4->0.1-0.15, both at the same
    # curriculum step) was trained for the full 15,000 iterations and MEASURED to
    # regress badly relative to this task's own prior documented baseline (0.260m
    # mean head height, 94.5% standing success): it plateaued by iteration ~3000-
    # 6000 at ~0.18m head height, ~0.70 upright error (well off vertical), and
    # ~60deg RMS joint deviation from home — and never moved from that plateau
    # for the remaining 9000+ iterations, i.e. a genuine stuck local optimum, not
    # an undertrained one. Root cause (inferred from which joints were most
    # off — shoulders/elbows 50-90deg, hips/ankles 55-95deg): opening the
    # pose-matching gate that early (~0.13m, about half of standing height) starts
    # pulling the arms back toward their home configuration while this robot still
    # needs to brace with its arms against the ground to complete the actual
    # stand-up motion, fighting the recovery itself rather than just polishing it.
    #
    # This version keeps only the low-risk part of that fix (per-joint MSE instead
    # of a sum-of-squares-then-exp kernel, so the gradient doesn't collapse when
    # several joints are simultaneously off) and is otherwise deliberately close to
    # the ORIGINAL, empirically-working design: gate still centered near the
    # standing threshold (moved from 0.230 to 0.85*HEAD_STANDING_HEIGHT = 0.221,
    # i.e. exactly HEAD_STANDING_THRESHOLD, so pose-matching turns on together with
    # standing_bonus/on_feet/foot_flat rather than a full ~40% window before them)
    # with a softer-but-still-late sharpness (0.02, vs the original's near-step
    # 0.005 and the regressing version's 0.05), a single std (no loose/tight
    # curriculum swap — that swap's own mid-training reward-scale jump is itself a
    # plausible destabilizer, independent of the gate timing), and NO weight
    # ramp — a fixed, modest weight from step 0, avoiding a second reward-scale
    # discontinuity. std is calibrated to be equivalent-to-slightly-looser than the
    # original sum-of-squares/std=1.0 kernel rather than tighter: for N joints each
    # off by the same amount, sum(err^2)/1.0 == mean(err^2)/std^2 when
    # std == 1/sqrt(N) ≈ 1/sqrt(18) ≈ 0.235 — used as the common value below
    # instead of per-joint-tuned values, so this change doesn't also (silently,
    # like the regressing version did) make the reward harder to earn than before.
    # 0.235 (calibrated to roughly match the OLD sum-of-squares/std=1.0 kernel's
    # tolerance) turned out to be far tighter than the actual joint errors a
    # standing-but-not-yet-home-postured policy has: measured directly (eval
    # script) that once genuinely standing, RMS joint deviation from home was still
    # ~53-65 degrees (~0.9-1.1 rad) — plugging that into
    # exp(-mean(error^2/std^2)) with std=0.235 numerically underflows to ~0,
    # meaning this reward (and its gradient) were effectively zero even while
    # weighted at 20.0, exactly like the head_height power>1 regression earlier:
    # technically-correct shape, but far too unforgiving for where training
    # actually is right now to provide any usable signal. Loosened to 0.6 (~34deg)
    # so a 50-60deg current error still gives a small-but-nonzero reward/gradient
    # to shrink from, rather than a numerical zero.
    HOME_POSE_STD = {r".*": 0.6}
    cfg.rewards["standing_pose"] = RewardTermCfg(
        func=home_pose_reward,
        # 240.0 is the FINAL weight, ramped up by pose_curriculum below (see that
        # curriculum term's own comment) — starts at 6.0 here. Measured directly
        # (a from-scratch run with every weight already at its final value):
        # standing_bonus/standing_pose plateau within the first ~5000 iterations
        # and never break past a partial-height, inconsistent-standing local
        # optimum through 20000 iterations, unlike this same reward SET reached
        # via many incremental resumes each starting from an already-mostly-
        # working policy. Full-strength pose-matching pressure competing with
        # standing_pose/hip_pose/head_height_sq all at once, before the robot has
        # even learned to reliably stand at all, looks like the cause — starting
        # low and ramping up once standing is reliable reproduces the effective
        # shape of that incremental history in one from-scratch run.
        weight=6.0,
        params={
            # Own threshold (0.8), lower/earlier than standing_bonus's (0.9) — pose
            # matching gets a head start on shaping before the "sustain full
            # height" bonus commits, since pose_reward's own exp(-error) shape
            # already provides continuous gradient once gated (no separate ramp
            # needed the way standing_bonus's flat-until-scaled version did).
            # Exactly 0 below the threshold, full pose_reward above it.
            "height_threshold": 0.8 * HEAD_STANDING_HEIGHT,
            "std": HOME_POSE_STD,
            "head_asset_cfg": HEAD_ASSET_CFG,
            "asset_cfg": SceneEntityCfg("robot", joint_names=(dofs_filter,)),
        },
    )

    # Dedicated hip pose-matching term, SEPARATE from standing_pose above (same
    # home_pose_reward class, reused, scoped to just the 6 hip joints via
    # asset_cfg: left/right x hip_yaw/hip_roll/hip_pitch — widened from just
    # hip_roll/hip_pitch (4 joints) after a live rollout showed the hips as
    # specifically where deviation still concentrates, hip_yaw included, not just
    # roll/pitch. Tried folding a tighter std for these joints INTO standing_pose's
    # shared std dict first and measured it backfire: standing_pose computes ONE
    # exp(-mean(error^2/std^2)) over all 18 joints, so a single joint with a huge
    # error (left_hip_pitch was ~98deg off) against a tight std dominates that mean
    # enough to crush the WHOLE term toward zero — including the gradient for the
    # OTHER joints that were already converging nicely. A separate term has its
    # own independent exp(), so a tight std here only affects the hip joints' own
    # reward/gradient, leaving standing_pose's (still at the looser uniform 0.6)
    # alone.
    cfg.rewards["hip_pose"] = RewardTermCfg(
        func=home_pose_reward,
        # 240.0 is the FINAL weight (equal to standing_pose's — a live rollout
        # showed the hips still the dominant source of remaining deviation even
        # after a weight increase broke the original ~11-12/~7.5-8 plateau, so
        # hip_pose gets matched up to standing_pose's own weight rather than
        # staying at some fraction of it), ramped by pose_curriculum below
        # alongside standing_pose — see that curriculum term's own comment for
        # the full weight-escalation history and why this starts low.
        weight=6.0,
        params={
            "height_threshold": 0.8 * HEAD_STANDING_HEIGHT,
            # 0.6, NOT tighter: left_hip_pitch's current error (~98deg/1.7rad) means
            # even std=0.3 numerically vanishes (exp(-(1.7/0.3)^2) underflows to
            # ~0) — same lesson as standing_pose's own std history, just re-applied
            # here. Start loose enough for real gradient at today's error, tighten
            # once it's actually shrunk.
            "std": {r".*": 0.6},
            "head_asset_cfg": HEAD_ASSET_CFG,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=(r".*hip_yaw.*", r".*hip_roll.*", r".*hip_pitch.*")
            ),
        },
    )

    # "Don't flail when picked up" (airborne_compliance_reward) is dropped along with
    # hold_airborne — with the triggering event disabled, this was only ever paying
    # out for the rare accidental airborne moment, not exercising the behavior it was
    # meant to train. See microban_teleop/docs/handling_research.md for the shelved
    # design if this gets revisited later.

    # Torque penalty starts at 0 (stage 1: find any way up) and ramps in via curriculum
    # once standing succeeds often (stage 2: keep it within what the real servos can
    # deliver — see microban_teleop/docs/getup_research.md on why this matters for the
    # XC330-T288-T's torque/current limits).
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(func=joint_torques_l2, weight=0.0)

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=velocity_mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_sensor_cfg.name},
    )

    #---------------------------- Terminations ----------------------
    # No early "fell over" exit — starting fallen is the whole point. Keep time_out and
    # out_of_terrain_bounds only.
    del cfg.terminations["fell_over"]

    # Safety net for hold_airborne (see events below): an external force applied to
    # the trunk on top of an already-extreme randomized fallen pose occasionally pushes
    # the MuJoCo solver into a genuine NaN/Inf blowup (reproduced directly: ~1 env per
    # few hundred steps out of 4096, even after staggering hold_airborne's first-trigger
    # timing and reducing its force range). A single NaN env otherwise aborts the whole
    # training run (rsl_rl's check_nan checks the full batch). Resetting just the
    # affected env here, before it corrupts anything else, is cheap and keeps training
    # running instead.
    cfg.terminations["nan_detection"] = TerminationTermCfg(func=envs_mdp.nan_detection)
    # See extreme_joint_velocity's docstring: catches unphysical-but-still-finite
    # blowups that precede (and, left unchecked, cause) the NaN nan_detection catches.
    cfg.terminations["extreme_joint_velocity"] = TerminationTermCfg(
        func=extreme_joint_velocity,
        params={"max_joint_vel": 50.0},
    )

    #---------------------------- Events ------------------------------
    # An initial-state (fallen pose) difficulty curriculum — start narrower, widen
    # via a step-count-gated curriculum stage — was tried and MEASURED to make
    # things worse, not better: performance dropped sharply the instant the range
    # widened (iteration 800) and never recovered for the remaining 14,000+
    # iterations, ending measurably worse (head height 0.044m, upright error 0.99)
    # than training on the full range from step 0 (head height 0.10-0.19m). A fixed
    # iteration count to widen on, uninformed by whether the policy has actually
    # mastered the easy regime yet, is a plausible reason: it forces a distribution
    # shift the policy hadn't earned readiness for. Reverted back to full-range
    # randomization from step 0.
    #
    # Full-range orientation (any lying pose) and joint angles (any limb
    # configuration, covering "powered on already on the ground"). position_range
    # is wider than any joint's own range so the post-clamp result saturates at
    # that joint's actual limits (reset_joints_by_offset clamps to
    # soft_joint_pos_limits).
    cfg.events["reset_base"].params["pose_range"] = {
        "x": (-0.1, 0.1),
        "y": (-0.1, 0.1),
        "z": (0.05, 0.25),
        "roll": (-3.14159, 3.14159),
        "pitch": (-3.14159, 3.14159),
        "yaw": (-3.14159, 3.14159),
    }
    cfg.events["reset_robot_joints"].params["position_range"] = (-3.14159, 3.14159)

    # Re-place a fraction of the just-randomized-fallen envs back into the home
    # pose (+ small noise) instead — found in FRASA (arXiv:2410.08655, Rhoban's own
    # fall-recovery policy: its "reset_final_p") and HumanUP (arXiv:2502.12152:
    # "standing_init_prob"/_reset_stand_and_lie_states), both independently, while
    # surveying the literature for the "ends in wrong posture" fix (see
    # home_pose_reward's docstring in mdp.py). Without this, home_pose_reward only
    # ever gets on-policy gradient from wherever the get-up motion happens to arrive
    # organically — it never directly practices "stay AT home", only "pass near home
    # once, at the tail of a long noisy rollout". rel_near_home_envs=0.1 matches
    # FRASA's own value. Relies on dict insertion order (this key is added after
    # reset_base/reset_robot_joints above) so it overrides their fallen-pose sample
    # for the selected envs rather than being overwritten by them.
    cfg.events["reset_near_home"] = EventTermCfg(
        mode="reset",
        func=reset_near_home_fraction,
        params={
            "rel_near_home_envs": 0.1,
            "joint_noise_range": (-0.05, 0.05),
            "orientation_noise_range": (-0.09, 0.09),  # ~+-5 deg roll/pitch
            # All joints (not dofs_filter-restricted): this is a physical state
            # reset, not the policy's action/observation space — head/neck should
            # also land near their (0.0) default for a "near home" episode rather
            # than keep whatever full-range angle reset_robot_joints gave them.
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    del cfg.events["push_robot"]  # no external pushes needed while learning to stand up

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
        r".*left_foot_collision.*",
        r".*right_foot_collision.*",
    )
    cfg.events["base_com"].params["asset_cfg"].body_names = ("trunk",)
    cfg.events["base_com"].params["ranges"] = {
        0: (-0.005, 0.005),
        1: (-0.005, 0.005),
        2: (-0.005, 0.005),
    }

    cfg.events["dof_armature_randomization"] = EventTermCfg(
        mode="startup",
        func=dr.joint_armature,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
            "operation": "scale",
            "ranges": (0.9, 1.1),
        },
    )
    cfg.events["dof_friction_randomization"] = EventTermCfg(
        mode="startup",
        func=dr.joint_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
            "operation": "scale",
            "ranges": (0.9, 1.1),
        },
    )

    # "Don't flail when picked up" (hold_airborne, simulating a hand lifting the
    # trunk) is shelved for now — parked here as a deliberate scope cut, not deleted:
    # combining it with get-up made the training signal harder to read (mid-episode
    # near-weightless trunk confusing to judge get-up quality against) and diluted
    # the reward the get-up motion itself needed (airborne_compliance_reward has since
    # been removed too, see below). The airborne gate on head_height_reward is left in
    # place (a harmless no-op without hold_airborne ever firing) so re-enabling this
    # later is still a small, contained change.
    #
    # cfg.events["hold_airborne"] = EventTermCfg(
    #     mode="step",
    #     func=hold_airborne,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=("trunk",)),
    #         "force_z_range": (8.0, 11.5),
    #         "force_lateral_range": (-0.5, 0.5),
    #         "torque_range": (0.0, 0.0),
    #         "duration_s": (1.0, 4.0),
    #         "cooldown_s": (3.0, 8.0),
    #     },
    # )

    #---------------------------- Curriculum ------------------------
    cfg.curriculum = {}
    cfg.curriculum["staged_curriculum"] = CurriculumTermCfg(
        func=step_based_staged_curriculum,
        params={
            "stages": [
                {
                    # -1e-4 (the original weight) proved too weak: the trained motion
                    # still looked violently forceful throughout the recovery, not just
                    # once standing (where standing_torque separately discourages it).
                    # 10x stronger to actually discourage that during the whole episode.
                    "name": "add torque regularization",
                    "step": 5000 * 24,
                    "apply": lambda env: env.reward_manager.get_term_cfg("joint_torques_l2").__setattr__(
                        "weight", -1e-3
                    ),
                },
            ],
        },
    )

    # Reproduces, in one from-scratch run, the effective shape of how this reward
    # set actually came together across many resumed sessions (each new
    # pose-precision term added/tuned only once the PRIOR stage had already
    # stabilized) — measured to matter: a from-scratch run with every weight
    # already at its final constant value plateaus at a partial-height,
    # inconsistent-standing local optimum through 20000 iterations and never
    # breaks past it. reward_based (not step-count-based, unlike staged_curriculum
    # above) since how many iterations it takes to first reliably stand is exactly
    # the kind of thing that varies run to run — advancing on "have you actually
    # reached this milestone" is more robust than guessing a fixed step count.
    cfg.curriculum["pose_curriculum"] = CurriculumTermCfg(
        func=reward_based_staged_curriculum,
        params={
            "stages": [
                {
                    # standing_pose/hip_pose start at 1/5th their final weight
                    # (see those reward terms' own comments) so full-strength
                    # pose-matching pressure doesn't compete with just learning to
                    # reliably stand at all in the first place. standing_bonus
                    # (max 5.0) plateaued at 0.03-0.96 for the full 20000
                    # iterations of the all-weights-final run — 2.0 is
                    # comfortably past that observed ceiling, so reaching it
                    # actually means "reliably standing", not just noise.
                    "name": "ramp up pose matching",
                    "reward_term_name": "standing_bonus",
                    "threshold": 2.0,
                    # Weight history, each measured directly before moving on:
                    # 30.0/15.0 (original target) plateaued hard at
                    # standing_pose~11-12 / hip_pose~7.5-8.4 for 400+ iterations
                    # with home_stillness held OFF (isolating this from any
                    # home_stillness interaction) — a genuine ceiling, not noise.
                    # 45.0/22.5 (1.5x) broke that, reaching standing_pose~17-18,
                    # but hip_pose's own scope was ALSO widened around the same
                    # time (4 hip_roll/hip_pitch joints -> 6, adding hip_yaw — see
                    # that term's own comment) and measured to only reach ~5 at
                    # that same 1.5x weight, not keeping pace with standing_pose.
                    # 60.0/60.0 (2x again) kept climbing (standing_pose~30-33,
                    # hip_pose~27-32) but still not converged onto home by the
                    # time the run neared its own iteration budget — doubled to
                    # 120.0/120.0, then close-but-not-quite on a live rollout at
                    # that weight too — doubled once more (240.0/240.0) along
                    # with stage 2's own thresholds below.
                    "apply": lambda env: (
                        env.reward_manager.get_term_cfg("standing_pose").__setattr__("weight", 240.0),
                        env.reward_manager.get_term_cfg("hip_pose").__setattr__("weight", 240.0),
                    ),
                },
                {
                    # home_stillness starts at 0.0 (fully off) — holding a
                    # commanded target still near home is a premature ask before
                    # the policy can even reliably MATCH that pose yet.
                    # Gated on BOTH standing_pose AND hip_pose (not standing_pose
                    # alone): with hip_pose's weight raised to match standing_pose
                    # (see that term's own comment — hips were still the dominant
                    # source of remaining deviation), a live rollout showed
                    # hip_pose lagging behind standing_pose's own climb; requiring
                    # both stops home_stillness engaging while hip_pose is still
                    # well behind. Per-term (not shared) thresholds: hip_pose's
                    # own scope (6 joints, including the harder-to-converge
                    # hip_yaw — see that term's comment) measured a meaningfully
                    # lower achievable ceiling than standing_pose's at the same
                    # weight, so gating both on one shared number was either too
                    # easy for standing_pose or unreachable for hip_pose.
                    "name": "enable home stillness",
                    "reward_term_name": ["standing_pose", "hip_pose"],
                    "threshold": [120.0, 60.0],
                    "apply": lambda env: env.reward_manager.get_term_cfg("home_stillness").__setattr__(
                        "weight", 50.0
                    ),
                },
            ],
        },
    )

    #---------------------------- Play mode -------------------------
    if play:
        cfg.curriculum = {}
        cfg.events["reset_base"].interval_range_s = (0.0, 0.0)
        cfg.observations["actor"].enable_corruption = False

    return cfg


MicrobanGetupRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
    wandb_project="mjlab_microban_getup",
    experiment_name="mjlab_microban_getup",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=15_000,
)
