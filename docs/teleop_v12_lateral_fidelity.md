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

The pushed loss is larger on the left commands (A, B) than on their mirror
images (M, MB): under the push, M and MB keep +0.05 to +0.06 m/s while A and B
drop to about 0.00 to +0.02 m/s. The probe gives M and MB the mirror image of
the `mixed_forward_left` targets, so both sides carry the same layout up to
mirroring. The gap is a left/right asymmetry of the policy, and the v1 term
does not change it. The crossed probe below locates it.

## Crossed left/right probe (2026-10-06, diagnostic only)

`vprobe_x.py` (scratchpad `lean_diag3/`) crosses three factors:

- **Command direction:** left (A, B) or right (M, MB).
- **Target layout:** L (`mixed_forward_left` as the evaluator gives it), R (its mirror image) or off (no hand or foot target).
- **HMD:** the evaluator's random moving HMD; the exact mirror of a paired left env's HMD trajectory (head and neck_roll negated); or HMD held neutral.

It uses held-out seeds 101-105, 4 reps and 300 steps, so each cell below has
40 rollouts. Entries are signed lateral velocity in m/s with the random HMD,
and falls are in parentheses.

| checkpoint | push | left cmd, L | left cmd, R | left cmd, off | right cmd, L | right cmd, R | right cmd, off |
| --- | --- | --- | --- | --- | --- | --- | --- |
| frozen walker (source only) | none | +0.077 | - | +0.067 | +0.083 | - | +0.086 |
| frozen walker (source only) | p30 | +0.056 (0) | - | +0.054 (0) | +0.054 (1) | - | +0.055 (1) |
| 7099 (pre-hand) | none | +0.063 | +0.064 | +0.070 | +0.090 | +0.097 | +0.085 |
| 7099 (pre-hand) | p30 | +0.038 (0) | +0.040 (0) | +0.055 (0) | +0.067 (1) | +0.071 (1) | +0.054 (0) |
| V2 (-16) 7500 | none | +0.051 | +0.050 | +0.072 | +0.067 | +0.072 | +0.080 |
| V2 (-16) 7500 | p30 | +0.031 (0) | +0.032 (0) | +0.057 (0) | +0.059 (4) | +0.063 (1) | +0.056 (0) |
| pose-release 9999 (gated) | none | -0.001 | -0.001 | +0.068 | +0.016 | +0.019 | +0.068 |
| pose-release 9999 (gated) | p30 | -0.125 (22) | -0.093 (14) | +0.051 (0) | +0.028 (9) | +0.032 (8) | +0.050 (1) |
| pose-release 14999 (`2026-10-06_05-26-14`) | none | +0.003 | +0.010 | +0.063 | +0.032 | +0.029 | +0.067 |
| pose-release 14999 (`2026-10-06_05-26-14`) | p30 | -0.124 (22) | -0.063 (15) | +0.046 (0) | +0.017 (4) | +0.024 (3) | +0.047 (0) |

On 14999 with layout L, the left/right values (no push, then p30) are:

- random HMD: +0.003 / +0.032, then -0.124 / +0.017
- mirrored HMD: -0.007 / +0.032, then -0.091 / +0.020
- neutral HMD: -0.009 / +0.023, then -0.078 / +0.020

What this shows:

- **Layout is not the cause.** On every checkpoint, swapping L and R changes little on either side.
- **HMD is not the cause.** Mirroring the HMD motion or holding it neutral leaves the gap in place.
- **The gap follows the command direction.** It already exists at 7099, before hands activate (no push +0.063 left vs +0.090 right). It is absent in the frozen walker.
- **Hand activation drives the main loss on both sides.** With targets active, lateral speed collapses on both sides from 7100 on, while forward speed rises (9999: vx about 0.28 against 0.09 to 0.13 at 7099). The weaker left side goes negative and falls under the push. With targets off, both sides keep +0.05 to +0.07.

## V3: revision v2, `hand_active_lateral_shortfall`

The v1 term's lateral target, `min(r * p_x, |c_y|)`, scales with the robot's
own forward progress, so a policy can lower it by walking forward more slowly
without stepping sideways. V1 and V2 did this: forward speed fell, and pushed
lateral speed on A and B stayed near 0. Revision
`hand_pose_release_lateral_fidelity_v2` (labels `s24` = -24, `s40` = -40)
replaces the term:

```
p_y = v_y * sgn(c_y)
shortfall = max(0, min(|c_y|, 0.10) - p_y)     if |c_y| >= 0.05 and a hand target is active, else 0
```

It never reads v_x or c_x. Pure-lateral commands with an active hand are
covered too. Because the shortfall is computed per env, it weights the weaker
command direction more by construction. The marker (schema 2) also records
`lateral_cap_m_s` and `requires_active_hand`. Validation now compares
canonical JSON, so a `True` in place of `1`, or an int in place of a float, is
drift.

The rules were written before training, in scratchpad `lean_diag3/status.txt`.
They differ from V1 and V2 as follows:

- **Larger probe.** Seeds 101-105 and 4 reps give 80 rollouts per pooled cell and 40 per side.
- **Later first checkpoint.** The first early-warning checkpoint is model_8000.
- **Same thresholds at every checkpoint.** These apply at 8000, 9000, 9999 (before its gate), 12000 and 14000:
  - no push, full targets: ≥ +0.050
  - p30, full targets: ≥ +0.030, with the left side ≥ +0.015 and ≤ 10 falls in 80
  - no push, hands off: ≥ +0.045
- **Fall cap restated.** The 14999 screen keeps the ≤ 10 of 80 p30 fall cap from the V1/V2 amendment.
- **Two-sided veto.** The veto checks the gate command G and its mirror image GM, both with the evaluator push, on seeds 101-105.

The final gate profile is unchanged. V3 may spend at most two more final-gate
tries (11 and 12).
