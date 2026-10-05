# Contract-v12 pose-release lateral-fidelity variant (forward-lean HOME)

## Why

The forward-lean pose-release chain (`lean_v12_pr_*`) failed the unchanged
15000 final gate
(`full_body_reachable_performance_perturbation_v2_completion_allowance_v1`) in
all 10 tries on `mixed_forward_left`: the twist (0.7, 0.3, 1.5) m/s, rad/s
with the evaluator push gave lateral velocity below the 0.02 m/s minimum, or
the robot fell. A held-out probe (seeds 101/102, commands (0.6, 0.3, 1.2),
(0.7, 0.3, 1.0) and their mirrors, no push and a 0.30/0.15 m/s push) showed
the loss starting when hands activate at update 7100. Pooled lateral velocity
with full targets fell from +0.078 m/s at 7099 to +0.018 m/s at 9500, while the
centered line stayed near +0.07 m/s.

## Recipe

`microban_teleop_v12_lateral_fidelity` keeps every pose-release term, stage,
push and sampler, and adds one reward term in the HOME-levelled frame,
`mixed_command_lateral_deficit`. It applies only when `|c_x| >= 0.05` and
`|c_y| >= 0.05`:

```
p_x = v_x * sgn(c_x),  p_y = v_y * sgn(c_y),  r = |c_y| / |c_x|
deficit = max(0, min(r * p_x, |c_y|) - p_y)
```

The term is penalised at weight -8 (V1) or -16 (V2, the declared fallback).
Extra lateral speed and pure-axis commands are never penalised.

The variant restarts the gated fresh-chain model_7099
(`2026-10-05_14-02-37_lean_v12_pr_7000_to7100`, sha256 `ee7ac215...`, gate
`cc193cdf...`):

```
scripts/train_microban_teleop_v12.sh resume 2026-10-05_14-02-37_lean_v12_pr_7000_to7100 \
  --hand-pose-release --lateral-fidelity [--lateral-fidelity-weight 16] \
  --agent.run-name lean_v12_lf_7100_to10000 --seed 42
```

How the marker works:

- **What it records.** Every save carries the `microban_teleop_v12_lateral_fidelity` marker: the revision `hand_pose_release_lateral_fidelity_v1`, the weight, the threshold, and the parent checkpoint and its stage gate by path and SHA-256. The recipe string stays the pose-release v18 string.
- **Who checks it.** The lineage check, the stage gates, the packager (`v12_lateral_fidelity_*` metadata) and the pose-release corner rescue all re-validate the marker. The robot (`microban_lean` `forward-lean-home`, `pico_hybrid.py`) refuses a package whose marker is partial or has drifted.
- **What the runner refuses.** It will not resume a marked checkpoint without the term at the recorded weight. It will not train the term on an unmarked checkpoint.
- **Corner rescue.** It carries the term automatically when its parent is marked.
- **Final rescue.** The pose-release final rescue does not accept lateral-fidelity parents.

## Result (2026-10-06): stopped by the pre-registered early-warning probe

Selection rules were written before training. The 7500 probe (seeds 101/102, 4 commands × 4 reps) required all three of these:

- no-push, full targets: ≥ +0.060 m/s
- p30 push, full targets: ≥ +0.040 m/s, with ≤ 6 falls
- no-push, hands off: ≥ +0.055 m/s

Lateral velocity by run at model_7500:

| run (model_7500) | no push, full | p30, full (falls / 32) | no push, hands off | forward vx (no push, full) |
| --- | --- | --- | --- | --- |
| plain pose-release `lean_v12_pr_7100_to10000` (reference) | +0.052 | +0.039 (1) | +0.074 | 0.197 |
| V1, weight -8 `2026-10-06_08-17-05_lean_v12_lf_7100_to10000` | +0.052 | +0.023 (3) | +0.075 | 0.198 |
| V2, weight -16 `2026-10-06_08-34-40_lean_v12_lf16_7100_to10000` | +0.0596 | +0.036 (2) | +0.076 | 0.180 |
| centered pose-release `c20k_v12_pr_7100_to10000` (reference) | +0.070 | +0.056 (1) | +0.077 | 0.118 |

Both variants failed, and as declared the chain stopped. Neither run was gated, and both carry `DO_NOT_USE_UNGATED.txt`. The lean 14999 final gate was not tried again, so the total stays at 10 tries, all failed.

The loss is one-sided. The mirrored right commands (M, MB) keep +0.05 to +0.06 m/s under the push. The left commands (A, B) carry the `mixed_forward_left` hand and foot targets, and they drop to about 0.00 to +0.02 m/s. The lateral term does not change this asymmetry.
