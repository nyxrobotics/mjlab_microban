"""A training floor that gives sideways like a carpet (shear compliance).

The floor is one plate under the robot, on two horizontal slide joints (x and y) with a spring and a damper on
each.  Each environment draws, once at startup:

* a spring k log-uniformly from 500 to 5000 N/m, the same in x and y: 2.7 N, the largest sideways foot force of
  standing and walking, bends it 5.4 to 0.54 mm;
* a damping ratio zeta uniformly from 0.3 to 1.0 for the mass that sways on the spring when the robot stands on
  the plate, the robot's and the plate's m = 2.136 kg: c = 2 zeta sqrt(k m), 20-65 N s/m at 500 N/m and 62-207 N s/m
  at 5000 N/m;
* whether its floor is rigid: one third of the environments keep the static ground.

The plate weighs 1 kg and travels +-10 mm.  It is that heavy because MuJoCo integrates the damper implicitly
(implicitfast), which keeps an unloaded plate stable at any damping, but its contact solver computes the robot's
contact forces with the plate's bare mass: under a contact force the plate then moves m / (m + dt c) as far as the
solver assumed.  On a 0.1 kg plate (dt c / m up to 7.5) a load resting on it is carried with the wrong damping and
slips: at 5000 N/m and zeta 0.3 a 1.136 kg box resting on the plate rang with a damping ratio of 0.002 and slid
11.6 mm on it, and the robot lying on it with its servos holding never settled (40-115 mm/s after 2.5 s, 1-6 mm/s on
the rigid floor; damping ratio 0.01).  The damping ratio of the robot lying on the stiffest, least damped plate
(0.3 drawn) came out 0.16, 0.21, 0.23 and 0.25 at 0.3, 0.5, 0.7 and 1 kg; at 1 kg (dt c / m <= 1.03) the box keeps
0.26-0.29 and 0.90-0.99 for 0.3 and 1.0 drawn over the whole spring range and slips at most 0.09 mm.  The plate's
free frequency sqrt(k / m) is at most 71 rad/s, under the 5 ms physics step's bound of 2 / dt = 400 rad/s for the
explicit spring.  A reset puts the plate back at its centre at rest (mjwarp's reset_data restores qpos0 and zero
velocity in the reset worlds).

Geometry.  MuJoCo allows a plane only on a static body, so the plate is a box, 10 m square and 0.1 m thick, its top
at the ground's height, centred under each environment's origin at every reset (the get-up stays near it, the
whole-body references within 1.9 m).  A walking robot goes farther (20 s at up to 0.7 m/s), so the plate's centre
moves under a robot FLOOR_PLATE_RECENTRE_M from it.  It is a child of the ground's body ``terrain``, so the contact sensors that match the
ground's subtree see both.  In MuJoCo Warp's float32 a box much larger loses its contacts: on a 20 m half-size
plate the standing robot's soles kept 23 of their 48 floor contacts and sank 0.21 mm instead of 0.07 mm; at 0.5 to
5 m the soles keep 47-48 contacts and sink 0.07 mm, as on the plane.  Where the robot lies on the floor, MuJoCo
Warp's box contacts (convex collision) still differ from the plane's (analytic): the same 12 get-up environments
with the same random actions for 8 s had 5.1 instead of 7.8 floor contacts per environment and step, sunk 0.82
instead of 0.34 mm at the median and 7.6 instead of 2.9 mm at the 95th percentile, the same on a locked plate and on
a 0.5 m one; the training's floor softness (time constant 0.02-0.1 s, sink growing with its square) spans more than
that.  So a rigid environment does not lock the plate but keeps the plane: its plate sits 1 m below the ground, out
of reach, and a compliant environment's plane sits 1 m below its plate.  The plane's height is set per environment in mjwarp's data: a static geom's pose is computed
once, when the data is made, and never again (mjwarp smooth.kinematics).

Contact softness.  MuJoCo regularizes a contact by its two bodies' inverse weights (body_invweight0), which for
a 0.1 kg plate (10 /kg, sliding sideways only) doubled the standing robot's sink on a compliant plate (0.07 to
0.15-0.17 mm at the 0.02 s contact time constant), a softening in the normal direction the plate does not have.  The
plate's inverse weight is set to 0 at every reset, so its contacts are as soft as the ground's; the plate's
sideways mobility stays in the dynamics.

Training only: evaluation and play keep the static ground.
"""

from __future__ import annotations

from dataclasses import replace

import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.lab_api.math import sample_log_uniform

GROUND = "terrain"  # mjlab's ground body and its plane geom
FLOOR_PLATE = "floor_plate"  # the plate's body and its box geom
FLOOR_PLATE_JOINTS = ("floor_plate_x", "floor_plate_y")
FLOOR_PLATE_MASS_KG = 1.0
FLOOR_PLATE_HALF_SIZE_M = (5.0, 5.0, 0.05)
FLOOR_PLATE_TRAVEL_M = 0.01
FLOOR_PLATE_STIFFNESS_N_M = (500.0, 5000.0)
FLOOR_PLATE_DAMPING_RATIO = (0.3, 1.0)
RIGID_FLOOR_FRACTION = 1.0 / 3.0
# How far below the floor the unused one of the ground plane and the plate sits.
UNUSED_FLOOR_DEPTH_M = 1.0
# A walking robot leaves the plate's centre (20 s at up to 0.7 m/s); the plate follows it from this far.
FLOOR_PLATE_RECENTRE_M = 2.0


def add_floor_plate(spec: mujoco.MjSpec) -> None:
    """Add the plate under the ground's body (a SceneCfg.spec_fn).

    The ground is the scene's first body, so the plate's joints come first in qpos: the scene's keyframe, the
    entities' joint positions, gets the plate's centre in front of it.
    """
    ground = spec.body(GROUND)
    (plane,) = ground.geoms
    plate = ground.add_body(name=FLOOR_PLATE, mass=FLOOR_PLATE_MASS_KG, inertia=[1.0e-4] * 3)
    plate.explicitinertial = True  # the plate only slides: its rotational inertia is never used
    for name, axis in zip(FLOOR_PLATE_JOINTS, ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), strict=True):
        plate.add_joint(
            name=name,
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=axis,
            range=(-FLOOR_PLATE_TRAVEL_M, FLOOR_PLATE_TRAVEL_M),
            limited=mujoco.mjtLimited.mjLIMITED_TRUE,
        )
    plate.add_geom(
        name=FLOOR_PLATE,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=FLOOR_PLATE_HALF_SIZE_M,
        pos=(0.0, 0.0, -FLOOR_PLATE_HALF_SIZE_M[2]),
        # The ground's contact parameters (the robot's geoms have priority over both).
        contype=plane.contype,
        conaffinity=plane.conaffinity,
        condim=plane.condim,
        priority=plane.priority,
        friction=plane.friction,
        solref=plane.solref,
        solimp=plane.solimp,
        margin=plane.margin,
        gap=plane.gap,
        material=plane.material,
        rgba=plane.rgba,
    )
    # The plane and the plate never touch (the unused one sits 1 m below the other; MuJoCo's parent filter
    # leaves pairs with a body welded to the world).
    spec.add_exclude(bodyname1=GROUND, bodyname2=FLOOR_PLATE)
    # Box-box contacts by MuJoCo's box collider, not the general one (native CCD): on the plate the general one
    # gives the feet's thin boxes fewer contacts than the plane does, drops the foot touch sensors, and makes a
    # standing whole-body policy shake (0.043 rad/s leg speed on a locked plate vs 0.0016 on the plane).
    spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    for key in spec.keys:
        key.qpos = [0.0] * len(FLOOR_PLATE_JOINTS) + list(key.qpos)


@requires_model_fields("body_pos", "jnt_stiffness", "dof_damping")
def floor_plate(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    stiffness_range: tuple[float, float],
    damping_ratio_range: tuple[float, float],
    rigid_fraction: float,
) -> None:
    """One floor per environment: ``rigid_fraction`` of them (rounded) the static ground, the others a plate
    centred under the origin with a spring k drawn log-uniformly and a damper of a damping ratio drawn uniformly
    for the robot's and the plate's mass, the same in x and y.  A rigid floor's plate has no spring or damper."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(env.device, torch.long)
    n = len(env_ids)
    mj_model = env.sim.mj_model
    plate_id = mj_model.body(FLOOR_PLATE).id
    # The mass that sways on the spring when the robot stands on the plate: the robot and the plate.
    mass = float(mj_model.body_mass[env.scene["robot"].indexing.body_ids.cpu().numpy()].sum())
    mass += float(mj_model.body_mass[plate_id])
    stiffness = sample_log_uniform(*stiffness_range, (n,), device=env.device)
    ratio = torch.empty(n, device=env.device).uniform_(*damping_ratio_range)
    damping = 2.0 * ratio * torch.sqrt(stiffness * mass)
    rigid = torch.zeros(n, dtype=torch.bool, device=env.device)
    rigid[torch.randperm(n, device=env.device)[: round(rigid_fraction * n)]] = True
    stiffness[rigid] = 0.0
    damping[rigid] = 0.0
    model = env.sim.model
    for name in FLOOR_PLATE_JOINTS:
        joint = mj_model.joint(name)
        model.jnt_stiffness[env_ids, joint.id] = stiffness
        model.dof_damping[env_ids, int(joint.dofadr[0])] = damping
    origins = env.scene.env_origins[env_ids]
    unused = torch.tensor([0.0, 0.0, UNUSED_FLOOR_DEPTH_M], device=env.device)
    model.body_pos[env_ids, plate_id] = origins - rigid[:, None] * unused
    # The ground plane is a static geom: mjwarp computes its pose once, so it is moved in the data.
    plane = mj_model.geom(GROUND).id
    env.sim.data.geom_xpos[env_ids, plane, 2] = torch.where(rigid, 0.0, -UNUSED_FLOOR_DEPTH_M)


@requires_model_fields("body_pos")
def floor_plate_follow(env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice | None) -> None:
    """Put the plate's centre under a robot more than FLOOR_PLATE_RECENTRE_M from it (every step): its top is flat,
    so the contacts do not change, and its spring and damper act on the joint displacement, which is kept."""
    plate_id = env.sim.mj_model.body(FLOOR_PLATE).id
    model = env.sim.model
    root = env.scene["robot"].data.root_link_pos_w[:, :2]
    centre = model.body_pos[:, plate_id, :2]
    far = torch.linalg.vector_norm(root - centre, dim=-1, keepdim=True) > FLOOR_PLATE_RECENTRE_M
    model.body_pos[:, plate_id, :2] = torch.where(far, root.to(centre.dtype), centre)


@requires_model_fields("body_pos")
def floor_plate_recentre(env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice | None) -> None:
    """Put the plate's centre back under the environment's origin, where the robot starts (every reset)."""
    if env_ids is None:
        env_ids = slice(None)
    plate_id = env.sim.mj_model.body(FLOOR_PLATE).id
    model = env.sim.model
    model.body_pos[env_ids, plate_id, :2] = env.scene.env_origins[env_ids, :2].to(model.body_pos.dtype)


@requires_model_fields("body_invweight0")
def floor_plate_contact_weight(env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice | None) -> None:
    """Regularize the plate's contacts as the static ground's: no inverse weight of its own (module docstring)."""
    if env_ids is None:
        env_ids = slice(None)
    env.sim.model.body_invweight0[env_ids, env.sim.mj_model.body(FLOOR_PLATE).id] = 0.0


def train_on_floor_plate(cfg: ManagerBasedRlEnvCfg) -> None:
    """Train ``cfg`` on the plate: the plate under the ground, one floor drawn per environment, and the ground's
    contact sensors matching the ground's subtree (the plane and the plate)."""
    if cfg.scene.spec_fn is not None:
        raise ValueError("the scene already has a spec_fn")
    cfg.scene.spec_fn = add_floor_plate
    ground = ContactMatch(mode="body", pattern=GROUND)
    cfg.scene.sensors = tuple(
        replace(sensor, secondary=replace(ground, mode="subtree"))
        if isinstance(sensor, ContactSensorCfg) and sensor.secondary == ground
        else sensor
        for sensor in cfg.scene.sensors
    )
    cfg.events["floor_plate"] = EventTermCfg(
        func=floor_plate,
        mode="startup",
        params={
            "stiffness_range": FLOOR_PLATE_STIFFNESS_N_M,
            "damping_ratio_range": FLOOR_PLATE_DAMPING_RATIO,
            "rigid_fraction": RIGID_FLOOR_FRACTION,
        },
    )
    cfg.events["floor_plate_contact_weight"] = EventTermCfg(func=floor_plate_contact_weight, mode="reset")
    cfg.events["floor_plate_recentre"] = EventTermCfg(func=floor_plate_recentre, mode="reset")
    cfg.events["floor_plate_follow"] = EventTermCfg(func=floor_plate_follow, mode="step")
