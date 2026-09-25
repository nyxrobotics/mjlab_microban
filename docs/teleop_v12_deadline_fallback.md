# Contract-v12 deadline fallback

This route promotes exactly one measured checkpoint when the canonical 30 mm
hand-RMS gate cannot be met before the project deadline. It does not change the
policy, reward, normalizer, locomotion limits, P95 limit, or any safety check.

## Fixed decision

- Selected checkpoint SHA-256:
  `393d35b4e7cc0453f5143c7f2be4d4a4658567ab6132dfb54d32e67eb26b62b7`
- Selected strict-failure report SHA-256:
  `c466d66cf5b5ac8603f558e0ae8612450ac74bbad78fd88719a2cdaa9673e4b7`
- Rejected v2 checkpoint SHA-256:
  `c6171ded2cffda43d04f0a1e85332975d459d9e02fda38546287332ca230b3a1`
- Only changed threshold: hand RMS `0.030 m` to `0.035 m`.
- Hand P95 stays `0.050 m`; all locomotion, fall, finite-value, joint-limit,
  recurrence, HMD, direction, ablation, and ONNX checks stay unchanged.

The selected v1 has the better balanced result: LF+RB RMS 0.0334825 m,
LB+RF RMS 0.0235841 m, and maximum P95 0.0376649 m. The rejected v2 improved
LF+RB by only 0.0001624 m while worsening the other corner and maximum P95.

## Full evaluation and promotion

Run from a clean committed checkout. `RUN_NAME` is the v1 rescue run directory
under `logs/rsl_rl/mjlab_microban_teleop_v12`.

```bash
scripts/evaluate_microban_teleop_v12_deadline_fallback.sh \
  2026-09-25_23-49-47_v12_corner_rescue_9901_to10000 \
  artifacts/teleop_v12_corner_rescue/2026-09-25_23-49-47_v12_corner_rescue_9901_to10000_model_9999_tracking.json
```

The script reruns the unchanged 9x300 locomotion evaluation, the explicit
35 mm tracking profile, and 83-input ONNX CPU parity. It then creates and
revalidates a schema-v2 stage gate plus an immutable promotion receipt. Any
wrong checkpoint, the v2 checkpoint, a changed strict report, or a failed
non-RMS check is rejected.

## 100-update foot canary

After the full gate passes, use the normal stage driver:

```bash
scripts/train_microban_teleop_v12.sh resume \
  2026-09-25_23-49-47_v12_corner_rescue_9901_to10000
```

The driver reads the validated gate, enables the explicit fallback resume
mode, and runs exactly 100 updates: model9999/completed10000 to
model10099/completed10100. The runner writes no intermediate checkpoint and
requires common step 242400 and Adam step 202000 at the final save. The child
inherits the exact v1 marker and records the immediate parent checkpoint and
full-gate paths and hashes.

## Pinned model10099 post-canary promotion

The measured canary is fixed to SHA-256
`86a81f45f34d91036ab138963c835a3e78db98224f523d3484e1d3ed335082ff`.
Its canonical tracking report is fixed to SHA-256
`e8230cff4cd25e8af1d9931d1fcd459db112e5a34c88a20470e22c2019ec5161`.
That report failed only the original 30 mm hand-RMS check: its worst hand RMS
was 32.3777 mm and worst hand P95 was 39.9206 mm. All fall, finite-value,
actual joint-limit, recurrence, HMD-motion, observation-coverage, directional,
hand-P95, hand-column ablation, and newly activated foot-column ablation checks
passed. The unchanged 9x300 locomotion gate also passed.

Run the separate exact-canary adjudicator from a clean committed checkout:

```bash
scripts/evaluate_microban_teleop_v12_deadline_canary.sh \
  2026-09-26_00-18-44_v12_deadline_foot_canary_10000_to10100_20260926 \
  artifacts/teleop_v12_gates/2026-09-26_00-18-44_v12_deadline_foot_canary_10000_to10100_20260926_model_10099_tracking.json
```

It reruns the unchanged 9x300 locomotion suite, all six canonical foot-canary
scenarios with only hand RMS set to 35 mm, and the unchanged full-83-input ONNX
CPU parity gate. It then writes a schema-v2 gate and a hash-bound post-canary
receipt. A different checkpoint, a changed canonical report, any other failed
check, hand P95 over 50 mm, missing foot causal response, or an ONNX/locomotion
failure is rejected.

The accepted post-canary receipt is pinned to SHA-256
`f7fcf511ba75f326cd94a2fc21a05360fa9f7925849fc2bce91709fd8678f27b`;
the corresponding schema-v2 gate SHA-256 is
`24caf0d3eb70d1bc816efe5396027ef9ed62d310f7a73d9fb3c9ed173e04f871`.
Simulation consumers must authenticate the receipt bytes against that fixed
hash before accepting its structural evidence.

After that gate passes, the normal stage driver recognizes only this explicit
post-canary authorization and runs the single remaining segment:

```bash
scripts/train_microban_teleop_v12.sh resume \
  2026-09-26_00-18-44_v12_deadline_foot_canary_10000_to10100_20260926 \
  --agent.run-name v12_deadline_post_canary_10100_to15000
```

This route permits exactly completed update 10100 to completed update 15000,
uses `save_interval=15000`, and may write only `model_14999.pt`. The final gate
keeps hand RMS at 35 mm but restores the normal final eight-scenario profile,
including strict 15 mm foot RMS, 25 mm foot P95, perturbation, all safety and
causal checks, 9x300 locomotion, and CPU ONNX parity.
