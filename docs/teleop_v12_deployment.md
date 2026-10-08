# PICO package (`export_teleop_v12_deployment`)

The PICO policy of a release is the last checkpoint of its one training run,
`model_8999` (9000 updates, `PICO_TOTAL_UPDATES` in
`mjlab_microban/schedules.py`).  It is judged once
(`scripts/retrain_all_for_home.py`, step pico): the 9x300 locomotion
evaluation, the tracking evaluation under the one final profile and the ONNX
parity gate, all with seed 42, whose reports `teleop_v12_stage create` binds
with the checkpoint into one gate file (schema 3).  The pipeline then
packages it:

```bash
uv run --locked python -m \
  mjlab_microban.scripts.export_teleop_v12_deployment \
  --checkpoint logs/rsl_rl/mjlab_microban_teleop_v12/<run>/model_8999.pt \
  --stage-gate <state>/pico_judgment/<run>_model_8999_gate.json \
  --output <state>/release/pico_teleop.onnx --force
```

`--force` only permits the final atomic rename to replace an existing file
after every check passed; on any failure the previous output is unchanged.
The packager, on CPU:

1. rebuilds and compares the gate with the current evaluator code, rehashing
   the checkpoint, the three reports and the gate ONNX; requires
   `status=pass`, the final profile, a completed-update count of
   `PICO_TOTAL_UPDATES` and the checkpoint named after the gate's iteration;
2. captures the checkpoint bytes, revalidates the frozen walker and its probe
   (bootstrap provenance) and the corrected bilateral site order, and refuses
   dry-run evidence unless it packages a dry run;
3. exports a fixed-shape `obs[1,81] -> actions[1,18]` float32 graph;
4. writes the robot's contract microban-policy-1 from the validated evidence
   (docs/policies.md): the HOME stamp and layout, the PICO foot targets and
   frame, the arm-target box and slew, the curriculum, the per-joint raw-action guard derived from the final tracking
   envelope, and the startup self-test: the final tracking rollouts' actor
   observations (possible states only) with the actor's deterministic output
   for each;
5. checks parity of PyTorch, ONNX `ReferenceEvaluator` and ONNX Runtime
   `CPUExecutionProvider` on a deterministic 64-sample corpus before and after
   the metadata is attached, runs the robot's self-test rule on the final
   file, and publishes with `os.replace` and a directory `fsync`.

It does not run or hash the robot's sources: the robot checks a release when
it is installed (`tools/validate_policies.py src/agents` and its tests) and at
every start (the self-test).

## Tracking profile

One profile, `arm_overlay_foot_perturbation_v1`, the same at every HOME: all
scenarios (low forward, low forward with both arms 70 deg forward, one arm
reaching out while standing, both keypoint corners with the arms out, both
feet, two mixed twists), the push perturbation, the target-column ablation of
the arm and foot columns, and these limits:

| limit | value |
| --- | --- |
| foot RMS | 0.05 m |
| foot P95 | 0.08 m |

The arms are driven from outside (the robot's `pico_arms`), so hand accuracy
is not judged here.

plus no falls, finite values, actual soft-limit overshoot <= 0.25 rad, raw-action
recurrence, forced HMD motion, observation coverage and the twist directional
response.  The limits are the same at every HOME.

Focused CPU-only tests:

```bash
uv run --locked --with pytest python -m pytest -q \
  tests/test_teleop_v12_deployment.py tests/test_teleop_v12_stage.py tests/test_pico_schedule.py
```
