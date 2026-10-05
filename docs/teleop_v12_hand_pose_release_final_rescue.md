# Contract-v12 pose-release final-scenario rescue (14901 -> 15000)

The forward-lean active-hand arm pose-release chain (`lean_v12_pr_*`, branch
`forward-lean-v2`) trains 10100 -> 15000 as one segment. Two retrains of that
segment from the same gated 10099 parent and the same seed (42) failed the
unchanged 14999 stage gate the same way (2026-10-05):

| run | failed checks | scenario evidence |
| --- | --- | --- |
| `2026-10-05_19-28-30_lean_v12_pr_10100_to15000` | `actual_soft_limits`, `twist_directional_response` | bounded_both_feet soft 0.0921 rad; mixed_forward_left lateral -0.0042 m/s |
| `2026-10-05_22-23-51_lean_v12_pr_10100_to15000` | `actual_soft_limits`, `twist_directional_response` | mixed_forward_left soft 0.0920 rad, lateral -0.0043 m/s |

(limits: soft-limit overshoot <= 0.0873 rad, lateral directional response
>= 0.02 m/s). Hand/foot accuracy and the 9x300 locomotion gate passed; the
centered-HOME final scores +0.12 m/s on the same lateral check.

The rescue keeps the pose-release `model_14900` of the failed run and runs the
last 99 PPO updates again with the failed evaluator scenarios replayed in a
share of the episodes. It is the canonical final rescue
(`microban_teleop_v12_final_rescue`) for the pose-release lineage. **No gate
profile, threshold or check changes**: the rescue's `model_14999` is judged by
the ordinary 14999 stage gate under
`full_body_reachable_performance_perturbation_v2_completion_allowance_v1`
(hand 0.045 / 0.08 m, foot 0.055 / 0.11 m, every safety, soft-limit,
twist-response, locomotion and ONNX check).

## Route

- parent: the unmarked pose-release `model_14900.pt` (completed 14901, Adam
  step 298020, all 20 adapter columns) of a fresh chain, optionally through the
  pose-release model9900 corner rescue; no recipe switch, no final marker;
- failed gate: the same run's `model_14999` tracking report under the
  unchanged final profile. It may fail only hand/foot accuracy,
  `actual_soft_limits` and `twist_directional_response`; completion, falls,
  finiteness, recurrence, HMD motion, observation coverage and ablation must
  pass. The validator recomputes every check from the per-scenario evidence,
  requires the report's checkpoint to be the parent run's `model_14999.pt`
  whose bytes still hash to the report, and lists the failing scenarios;
- mix: every failing scenario must be replayed by the selected mix;
- replay: exactly 99 updates, 2048 environments, 24 steps, to
  `model_14999.pt` (completed 15000, Adam step 300000); training seed 42 by
  default, `--seed N` for another draw (training randomness only, recorded in
  the marker; every gate keeps its own fixed evaluation seeds).

## Mixes

A scenario is drawn once per episode and shared by the twist, foot and hand
commands (the evaluator holds one scenario for 300 steps): the replayed
episodes use the evaluator's twist, both foot targets, both hand targets and
hand-active flags exactly; the rest keep the ordinary samplers.

| mix | ordinary | mixed_forward_left | bounded_both_feet | mixed_backward_right |
| --- | --- | --- | --- | --- |
| `pr_v1` | 50 % | 30 % | 20 % | - |
| `pr_v2` | 70 % | 20 % | 10 % | - |
| `pr_v3` | 30 % | 45 % | 25 % | - |
| `pr_v4` | 50 % | 25 % | 15 % | 10 % |

`bounded_both_feet` lifts both feet (0.016 m, above the ordinary two-foot
range of 0.012 m) with both hands inactive; its replay uses the ordinary
stationary two-foot regime (`is_both_feet_env`). The centered v1 final rescue
at 10 % ordinary lost push robustness, so every mix keeps at least 30 %.

## Lineage

Saves keep the pose-release recipe revision and the inherited corner marker
and add a pose-release final-rescue marker under
`microban_teleop_v12_final_scenario_rescue` (revision
`recorded_pose_release_model14900_failed_final_scenarios_replay_99_updates_forward_lean_v1`).
It records the parent SHA-256, the failed gate (checkpoint and report
SHA-256, profile, failed checks and scenarios), the inherited corner-marker
SHA-256, the mix, its probabilities and the exact replayed commands. Every
consumer rebuilds it from those values (`hand_pose_release_lineage`):

- only the rescue's `model_14999` is consumable (the intermediate
  `model_14949` save is refused), lineage
  `fresh_chain_model9900_corner_rescue_model14900_final_rescue` (or
  `fresh_chain_model14900_final_rescue` without a corner rescue);
- the marker must name exactly the checkpoint's corner marker;
- the marker is refused on any other recipe, and the canonical final-rescue
  marker is refused on the pose-release recipe.

The rescue's 14999 stage gate records the rebuilt marker verbatim under the
same key (next to the corner marker), and `validate_gate` refuses a gate that
omits it, names a different one, or names one its checkpoint does not carry.
The package adds `v12_final_rescue_marker_revision`,
`v12_final_rescue_marker_json` and `v12_final_rescue_marker_sha256`
(canonical sorted-key JSON SHA-256) after checking the gate's marker equals
the checkpoint's, so an installed `pico_teleop.onnx` names its parent, the
failed gate it replayed, the mix and the seed. An ordinary final has none of
these fields.

The package otherwise declares the pose-release recipe and the
completion-allowance profile exactly like an ordinary pose-release final, so
the robot runtime (`microban_lean` `src/moves/pico_hybrid.py`,
`tools/validate_pico_policy.py`) needs no new table entry; it reads no
lineage fields and has no exact metadata key-set check.

The launcher stages a copy of the parent run's `params/agent.yaml` next to the
staged parent, so the packager's resume-ancestry walk continues from the seed
to the gated 10100 canary and 10000 boundary and records both gates.

## Launch and gate

From a clean commit:

```bash
scripts/train_microban_teleop_v12_hand_pose_release_final_rescue.sh \
  logs/rsl_rl/mjlab_microban_teleop_v12/<RUN>/model_14900.pt \
  artifacts/teleop_v12_gates/<RUN>_model_14999_tracking.json \
  --mix pr_v1 [--seed N] --agent.run-name lean_v12_pr_final_rescue_pr_v1_14901_to15000
scripts/evaluate_microban_teleop_v12_stage.sh <RESCUE_RUN> 14999
```

The launcher validates both inputs on CPU
(`python -m mjlab_microban.scripts.teleop_v12_hand_pose_release_final_rescue
validate-parent MODEL_14900 REPORT --mix pr_vN`), stages the parent, the
report, the parent run's resume record and `parent_run.txt` under
`pr_final_rescue_seed_<sha16>/`, and trains
`Mjlab-Teleop-V12-HandPoseRelease-Final-Rescue-Microban` (mix read from
`MICROBAN_V12_PR_FINAL_RESCUE_MIX`). Every runner load re-reads
`parent_run.txt`, requires the parent run's own `model_14900.pt` beside the
seed to hold the staged bytes, and re-validates the staged report against
that run (its `model_14999` in the same directory, bytes unchanged), so a
hand-staged seed cannot skip the same-run check. A passing gate is packaged and installed
with the ordinary finalize procedure using the rescue run name.
