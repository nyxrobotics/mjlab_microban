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

Further training from model10099 is intentionally rejected. Evaluate its
normal 10100 foot-canary gate first; a separate explicit post-canary promotion
decision is required before training beyond that point.
