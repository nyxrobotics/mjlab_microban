# Legacy locomotion actor in the 83-value teleop task

This probe answers one narrow deadline question: can the proven
`model_14999.pt` locomotion actor keep its original gait while the simulator is
the nominal `Mjlab-Teleop-Microban` task and exposes the 83-value PICO policy
observation?

The answer is **yes for neutral foot/hand targets and a HOME neck**. This is a
simulator feasibility result, not evidence for moving hand/foot retargeting or
real-hardware safety.

## Reproduce

```bash
cd /home/kanade/Git-projects/mjlab_microban_v8j
uv run --locked python \
  -m mjlab_microban.scripts.probe_legacy_actor_in_teleop_env \
  --steps 300 \
  --settle-steps 50 \
  --output artifacts/legacy_teleop_probe/model_14999_teleop83_raw_9x300.json \
  --force

uv run --locked --with pytest python -m pytest -q \
  tests/test_legacy_teleop_probe.py
```

The recorded receipt is
`artifacts/legacy_teleop_probe/model_14999_teleop83_raw_9x300.json`, SHA-256
`f51378d59ff4d68fb1185a91eb2a863749e5c7be6ec4cd0ab4a0b08f1565e69d`.
`artifacts/` is intentionally ignored by Git; the script and this procedure are
the reproducible source of the receipt.

## Measured result

- 9/9 scenarios completed all 300 steps (6 s each).
- 0 falls, 0 non-finite scenarios, and 0 measured joint soft-limit violations.
- All 8 nonzero signed-axis commands produced the requested response sign.
- The raw actor action reached the simulator unchanged on all 2,700 steps.
- The next observation contained that same raw action on all 2,700 steps.
- Foot and hand targets stayed exact-zero/inactive on all 2,700 steps.
- The three HMD joints stayed near HOME: maximum position departure was
  `0.0623238 rad` (`3.57 deg`) and maximum speed was `0.0218129 rad/s`.

The measured response means, compared with the existing original-task receipt,
were:

| Command | Teleop mapped actor | Original-task receipt |
|---|---:|---:|
| `vx=+0.1 m/s` | `+0.09259 m/s` | `+0.09669 m/s` |
| `vx=+0.2 m/s` | `+0.15848 m/s` | `+0.15873 m/s` |
| `vx=-0.1 m/s` | `-0.05454 m/s` | `-0.05346 m/s` |
| `vx=-0.2 m/s` | `-0.11121 m/s` | `-0.10608 m/s` |
| `vy=+0.1 m/s` | `+0.04311 m/s` | `+0.04282 m/s` |
| `vy=-0.1 m/s` | `-0.04589 m/s` | `-0.05625 m/s` |
| `yaw=+0.5 rad/s` | `+0.47998 rad/s` | `+0.48337 rad/s` |
| `yaw=-0.5 rad/s` | `-0.51913 rad/s` | `-0.52193 rad/s` |

The largest mean-response change was `0.01037 m/s` on right lateral motion; no
response reversed sign or lost completion. Both runs use the current 21-joint
robot model, so this comparison proves compatibility with the neck present at
HOME, but it does not reconstruct a historical pre-neck physics model and
therefore does not isolate the neck mass as a causal variable.

The actor sometimes requests an absolute target outside the current soft-limit
interval (maximum hypothetical excess `2.32737 rad`), as it also does in its
original contract. The measured joints did not cross their soft limits in this
probe. Applying the teleop target clip is not equivalent: the separate
soft-clipped diagnostic destroyed most locomotion and fell in backward motion.

## Exact 63-to-83 observation mapping

The mapping is resolved from joint names at runtime and then checked against
the following concrete columns:

| Legacy source | Teleop target | Meaning |
|---|---|---|
| `0:6` | `0:6` | angular velocity and projected gravity |
| `6:24` | `9:27` | 18 body joint positions |
| `24:42` | `30:48` | 18 body joint velocities |
| `42:60` | `48:66` | previous raw 18-action output |
| `60:63` | `66:69` | requested twist |

The new target columns are `6:9` (head/neck positions), `27:30` (head/neck
velocities), and `69:83` (six foot-target plus eight hand-target values).

## Exact actor transplant for v12

The executable transplant helper is
`transplant_legacy_actor_state_to_teleop83()` in the probe script. A compatible
v12 actor must start with these semantics:

- input width 83; hidden widths `512, 256, 128`; ELU; output width 18;
- `EmpiricalNormalization` enabled;
- the original unbounded `GaussianDistribution`, scalar standard-deviation
  parameterization;
- raw actor outputs passed unchanged to the joint-position action term;
- no action target clip; and
- the next actor observation contains the unchanged raw previous action.

Transplant the source actor only:

1. Copy the source normalizer mean, variance, and standard deviation into the
   63 mapped target columns. Initialize the 20 new columns to mean 0, variance
   1, and standard deviation 1. Copy the scalar normalizer count exactly
   (`1,474,560,000`).
2. Zero the target `mlp.0.weight` (`512 x 83`), then copy each source
   `mlp.0.weight[:, source_column]` into its mapped target column.
3. Copy `mlp.0.bias` and every later MLP weight/bias exactly.
4. Copy `distribution.std_param` exactly. Do not convert it to the bounded
   distribution or to log-standard-deviation coordinates.
5. Initialize the critic and optimizer fresh. Do not copy the velocity critic,
   optimizer, iteration counter, or PPO rollout state.

Because the new first-layer columns are zero, the initial actor ignores HMD and
foot/hand targets but can acquire them through gradient updates. The very large
copied normalizer count effectively freezes the new columns at identity
normalization; either freeze this normalizer deliberately during the
gait-preservation stage or introduce a separately audited per-column
normalizer. Resetting the shared count would immediately perturb the proven 63
columns and is not equivalent.

An in-memory transplant of the real checkpoint was compared over 1,024 random
83-value observations. Deterministic action parity had maximum absolute error
`3.8147e-6`, mean absolute error `2.45794e-7`, and the distribution
`std_param` was bit-exact. The focused CPU test suite also verifies mapping,
zero new first-layer weights, identity normalization of new columns, and strict
failure on an incomplete mapping.
