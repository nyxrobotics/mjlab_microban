"""Left/right mirror augmentation for get-up PPO (rsl_rl ``symmetry_cfg``).

rsl_rl 5.x ``PPO`` accepts ``symmetry_cfg={"use_data_augmentation": bool,
"use_mirror_loss": bool, "data_augmentation_func": <callable or
"module:attr">, "mirror_loss_coeff": float}`` and calls
``data_augmentation_func(env=<RslRlVecEnvWrapper>, obs=<TensorDict|None>,
actions=<Tensor|None>)``. Each non-None input comes back with its batch
doubled, original first then mirrored: ``cat([x, mirror(x)])``.

Mirror maps (derived from robot.xml and checked with MuJoCo kinematics, see
tests/test_getup_symmetry.py):

* The sagittal plane is the trunk x-z plane (the lateral axis is trunk +y).
  Body COMs and collision-geom centres of mirrored joint configurations
  reflect to within 0.96 mm and 0.05 mm. The 0.96 mm is a static 0.5 mm
  lateral COM offset of ``roll_pitch_link``.
* Joint ``right_X`` <-> ``left_X``, with ``q'[X] = sign * q[partner]``:
  -1 for yaw/roll axes (hip_yaw, hip_roll, ankle_roll, shoulder_roll, head,
  neck_roll) and +1 for pitch-type axes (shoulder_pitch, elbow, hip_pitch,
  knee, ankle_pitch, neck_pitch). HOME/default satisfies
  ``default = sign * default[partner]`` exactly, so joint positions relative to
  the default, joint velocities and default-relative actions all mirror with
  the same map.
* Trunk-frame polar vectors (projected gravity, root lin vel): (+1, -1, +1).
  Trunk-frame pseudo-vectors (root ang vel): (-1, +1, -1).
* Built-in site sensors are signed from the site's orientation in the trunk:
  ``diag(+-R_s^T M R_s)``, minus for gyros. The IMU site
  (quat 0.5,-0.5,-0.5,0.5) gives gyro ``robot/imu_ang_vel`` (+1, -1, -1) and
  velocimeter ``robot/imu_lin_vel`` (-1, +1, +1).

Every observation term is matched on its function and parameters. A term the
module does not recognise raises an error instead of passing through unmirrored.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch

# Lateral (mirror-normal) axis of the trunk frame, and the reflection matrix.
TRUNK_LATERAL_AXIS = 1
_M_TRUNK = np.diag([1.0, -1.0, 1.0])

# Sign of q'[joint] = sign * q[partner(joint)], by joint name without side prefix.
MIRROR_JOINT_SIGN: dict[str, int] = {
    "head": -1,
    "neck_roll": -1,
    "neck_pitch": +1,
    "shoulder_pitch": +1,
    "shoulder_roll": -1,
    "elbow": +1,
    "hip_yaw": -1,
    "hip_roll": -1,
    "hip_pitch": +1,
    "knee": +1,
    "ankle_pitch": +1,
    "ankle_roll": -1,
}

POLAR_TRUNK_SIGN = (1.0, -1.0, 1.0)
AXIAL_TRUNK_SIGN = (-1.0, 1.0, -1.0)


def mirror_joint_partner(name: str) -> tuple[str, int]:
    """Return (partner joint name, sign) for a joint name (entity prefix allowed)."""
    prefix, _, base = name.rpartition("/")
    prefix = prefix + "/" if prefix else ""
    if base.startswith("right_"):
        core, other = base[len("right_"):], "left_"
    elif base.startswith("left_"):
        core, other = base[len("left_"):], "right_"
    else:
        core, other = base, ""
    if core not in MIRROR_JOINT_SIGN:
        raise KeyError(f"No mirror sign known for joint {name!r}")
    return prefix + other + core, MIRROR_JOINT_SIGN[core]


def joint_mirror_perm_sign(names: list[str] | tuple[str, ...]) -> tuple[list[int], list[float]]:
    """Permutation/sign so that mirrored[i] = sign[i] * x[perm[i]] over ``names``."""
    index = {n: i for i, n in enumerate(names)}
    perm, sign = [], []
    for n in names:
        partner, s = mirror_joint_partner(n)
        if partner not in index:
            raise ValueError(f"Mirror partner {partner!r} of {n!r} is not in the term's joint list")
        perm.append(index[partner])
        sign.append(float(s))
    return perm, sign


def site_sensor_sign(mj_model: mujoco.MjModel, sensor_name: str, root_body: str = "trunk") -> tuple[float, float, float]:
    """Per-axis mirror sign of a 3-D site sensor expressed in its site frame."""
    sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sid < 0:
        raise KeyError(f"Sensor {sensor_name!r} not in model")
    stype = mj_model.sensor_type[sid]
    if mj_model.sensor_objtype[sid] != mujoco.mjtObj.mjOBJ_SITE:
        raise ValueError(f"Sensor {sensor_name!r} is not attached to a site")
    site = mj_model.sensor_objid[sid]
    body = mj_model.site_bodyid[site]
    body_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body)
    if body_name.rpartition("/")[2] != root_body:
        raise ValueError(f"Sensor {sensor_name!r} site is on {body_name!r}, not the {root_body!r} root body")
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, mj_model.site_quat[site])
    rot = rot.reshape(3, 3)
    pseudo = {int(mujoco.mjtSensor.mjSENS_GYRO)}
    polar = {
        int(mujoco.mjtSensor.mjSENS_VELOCIMETER),
        int(mujoco.mjtSensor.mjSENS_ACCELEROMETER),
    }
    if int(stype) in pseudo:
        m = -rot.T @ _M_TRUNK @ rot
    elif int(stype) in polar:
        m = rot.T @ _M_TRUNK @ rot
    else:
        raise ValueError(f"Sensor {sensor_name!r} type {int(stype)} has no known mirror rule")
    if np.abs(m - np.diag(np.diag(m))).max() > 1e-6:
        raise ValueError(f"Sensor {sensor_name!r} axes are not aligned with the trunk mirror axes")
    return tuple(float(round(v)) for v in np.diag(m))  # type: ignore[return-value]


@dataclass
class _GroupMirror:
    perm: torch.Tensor
    sign: torch.Tensor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.perm] * self.sign


def _term_perm_sign(env, term_name: str, term_cfg, dim: int) -> tuple[list[int], list[float]]:
    """Mirror perm/sign for one observation term of width ``dim`` (local indices)."""
    from mjlab.envs import mdp as envs_mdp

    from mjlab_microban.tasks.microban_getup_action import raw_getup_action

    func = term_cfg.func
    fname = getattr(func, "__name__", str(func))
    robot = env.scene["robot"]

    if fname == "builtin_sensor":
        sensor_name = term_cfg.params["sensor_name"]
        base_perm, base_sign = [0, 1, 2], list(site_sensor_sign(env.sim.mj_model, sensor_name))
    elif func is envs_mdp.projected_gravity or fname == "projected_gravity":
        base_perm, base_sign = [0, 1, 2], list(POLAR_TRUNK_SIGN)
    elif fname == "base_lin_vel":
        base_perm, base_sign = [0, 1, 2], list(POLAR_TRUNK_SIGN)
    elif fname == "base_ang_vel":
        base_perm, base_sign = [0, 1, 2], list(AXIAL_TRUNK_SIGN)
    elif fname in ("joint_pos_rel", "joint_vel_rel", "joint_pos", "joint_vel"):
        asset_cfg = term_cfg.params["asset_cfg"]
        ids = asset_cfg.joint_ids
        all_names = list(robot.joint_names)
        names = all_names if isinstance(ids, slice) else [all_names[i] for i in ids]
        base_perm, base_sign = joint_mirror_perm_sign(names)
        if fname == "joint_pos":
            raise ValueError("Absolute joint_pos needs an affine mirror; use joint_pos_rel")
        if fname == "joint_pos_rel":
            _check_default_symmetric(robot, names)
    elif func is raw_getup_action or fname in ("last_action", "raw_getup_action"):
        action_name = term_cfg.params.get("action_name", "joint_pos")
        term = env.action_manager.get_term(action_name)
        base_perm, base_sign = joint_mirror_perm_sign(list(term.target_names))
        _check_default_symmetric(robot, list(term.target_names))
    else:
        raise ValueError(f"No mirror rule for observation term {term_name!r} ({fname})")

    n = len(base_perm)
    if dim % n:
        raise ValueError(f"Term {term_name!r} width {dim} is not a multiple of its mirror width {n}")
    perm, sign = [], []
    for h in range(dim // n):  # flattened history: [h0 d..., h1 d..., ...]
        perm.extend(h * n + p for p in base_perm)
        sign.extend(base_sign)
    return perm, sign


def _check_default_symmetric(robot, names: list[str]) -> None:
    all_names = list(robot.joint_names)
    default = robot.data.default_joint_pos
    perm, sign = joint_mirror_perm_sign(names)
    ids = torch.tensor([all_names.index(n) for n in names], device=default.device)
    d = default[:, ids]
    mirrored = d[:, perm] * torch.tensor(sign, device=d.device, dtype=d.dtype)
    err = (mirrored - d).abs().max().item()
    if err > 1e-5:
        raise ValueError(f"Default joint pose is not mirror-symmetric (max {err:.3g} rad)")


def build_group_mirrors(env) -> dict[str, _GroupMirror]:
    """Build a mirror for every concatenated observation group, from the env's own term layout."""
    om = env.observation_manager
    mirrors: dict[str, _GroupMirror] = {}
    for group, term_names in om.active_terms.items():
        if not om.group_obs_concatenate[group]:
            raise ValueError(f"Observation group {group!r} is not concatenated")
        perm: list[int] = []
        sign: list[float] = []
        offset = 0
        for term_name, term_dim in zip(term_names, om.group_obs_term_dim[group], strict=True):
            if len(term_dim) != 1:
                raise ValueError(f"Term {group}/{term_name} is not 1-D: {term_dim}")
            dim = int(term_dim[0])
            term_cfg = om.get_term_cfg(group, term_name)  # resolved copy (joint_ids filled in)
            p, s = _term_perm_sign(env, term_name, term_cfg, dim)
            perm.extend(offset + i for i in p)
            sign.extend(s)
            offset += dim
        dev = env.device
        mirrors[group] = _GroupMirror(
            perm=torch.tensor(perm, dtype=torch.long, device=dev),
            sign=torch.tensor(sign, dtype=torch.float32, device=dev),
        )
    return mirrors


def build_action_mirror(env, action_name: str = "joint_pos") -> _GroupMirror:
    term = env.action_manager.get_term(action_name)
    perm, sign = joint_mirror_perm_sign(list(term.target_names))
    _check_default_symmetric(env.scene["robot"], list(term.target_names))
    return _GroupMirror(
        perm=torch.tensor(perm, dtype=torch.long, device=env.device),
        sign=torch.tensor(sign, dtype=torch.float32, device=env.device),
    )


# Cached on the env object itself: an id()-keyed module dict could hand a new
# env (allocated at a freed env's address) the old env's maps.
_CACHE_ATTR = "_getup_symmetry_mirrors"


def _mirrors_for(env) -> tuple[dict[str, _GroupMirror], _GroupMirror]:
    base = getattr(env, "unwrapped", env)
    cached = getattr(base, _CACHE_ATTR, None)
    if cached is None:
        cached = (build_group_mirrors(base), build_action_mirror(base))
        setattr(base, _CACHE_ATTR, cached)
    return cached


def mirror_obs_group(env, group: str, x: torch.Tensor) -> torch.Tensor:
    return _mirrors_for(env)[0][group](x)


def mirror_actions(env, a: torch.Tensor) -> torch.Tensor:
    return _mirrors_for(env)[1](a)


def getup_symmetry_augmentation(env, obs=None, actions=None):
    """rsl_rl ``data_augmentation_func``: returns (cat[obs, mirror obs], cat[actions, mirror actions])."""
    group_mirrors, action_mirror = _mirrors_for(env)
    obs_aug = None
    if obs is not None:
        mirrored = obs.clone()
        for key in obs.keys():
            if key not in group_mirrors:
                raise KeyError(f"No mirror for observation group {key!r}")
            mirrored[key] = group_mirrors[key](obs[key])
        obs_aug = torch.cat([obs, mirrored], dim=0)
    act_aug = None
    if actions is not None:
        act_aug = torch.cat([actions, action_mirror(actions)], dim=0)
    return obs_aug, act_aug


AUGMENTATION_FUNC_PATH = "mjlab_microban.tasks.microban_getup_symmetry:getup_symmetry_augmentation"


def make_getup_symmetry_cfg(
    use_data_augmentation: bool = True,
    use_mirror_loss: bool = False,
    mirror_loss_coeff: float = 0.0,
) -> dict:
    """``symmetry_cfg`` dict for rsl_rl ``PPO`` (the function is given as a string path)."""
    return {
        "use_data_augmentation": use_data_augmentation,
        "use_mirror_loss": use_mirror_loss,
        "data_augmentation_func": AUGMENTATION_FUNC_PATH,
        "mirror_loss_coeff": mirror_loss_coeff,
    }


# ---------------------------------------------------------------------------
# Config plumbing (not wired into any registered task).
#
# mjlab's RslRlPpoAlgorithmCfg has no ``symmetry_cfg`` field, and train.py does
# ``asdict(cfg.agent)``. This subclass adds a typed field; asdict() turns it
# into the plain dict rsl_rl's PPO(**cfg["algorithm"]) expects, it is recorded
# in params/agent.yaml, and tyro exposes it on the CLI as
# --agent.algorithm.symmetry-cfg.{use-data-augmentation,use-mirror-loss,mirror-loss-coeff}.
# ---------------------------------------------------------------------------
from dataclasses import dataclass as _dataclass, field as _field, fields as _fields, replace as _replace  # noqa: E402

from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg  # noqa: E402


@_dataclass
class GetupSymmetryCfg:
    use_data_augmentation: bool = True
    """Train on [batch; mirrored batch] (HumanUP/HiFAR-style augmentation)."""
    use_mirror_loss: bool = False
    """Add mirror_loss_coeff * MSE(pi(mirror(o)), mirror(pi(o))) to the PPO loss."""
    data_augmentation_func: str = AUGMENTATION_FUNC_PATH
    mirror_loss_coeff: float = 0.0


@_dataclass
class RslRlPpoSymmetryAlgorithmCfg(RslRlPpoAlgorithmCfg):
    symmetry_cfg: GetupSymmetryCfg = _field(default_factory=GetupSymmetryCfg)
    """rsl_rl symmetry_cfg. Non-optional so tyro exposes its fields as flags;
    with both use_* False rsl_rl only logs the mirror loss."""


def with_getup_symmetry(
    rl_cfg: RslRlOnPolicyRunnerCfg,
    use_data_augmentation: bool = True,
    use_mirror_loss: bool = False,
    mirror_loss_coeff: float = 0.0,
) -> RslRlOnPolicyRunnerCfg:
    """Copy of ``rl_cfg`` whose PPO uses the get-up mirror augmentation."""
    alg = rl_cfg.algorithm
    base = {f.name: getattr(alg, f.name) for f in _fields(RslRlPpoAlgorithmCfg)}
    sym_alg = RslRlPpoSymmetryAlgorithmCfg(
        **base,
        symmetry_cfg=GetupSymmetryCfg(
            use_data_augmentation=use_data_augmentation,
            use_mirror_loss=use_mirror_loss,
            mirror_loss_coeff=mirror_loss_coeff,
        ),
    )
    return _replace(rl_cfg, algorithm=sym_alg)
