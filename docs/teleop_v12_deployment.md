# PICO package (`export_teleop_v12_deployment`)

The PICO policy of a release is the checkpoint its one training run ended
with: model `TOTAL - 1` (9000 updates, `mjlab_microban/schedules.py`), or the
earlier checkpoint the pipeline's checks adopted once the last stage had run
at least half its length (`PICO_MIN_FINAL_UPDATES`, 7500).  It is judged once
(`scripts/retrain_all_for_home.py`, step pico): the 9x300 locomotion
evaluation, the tracking evaluation under the one final profile and the ONNX
parity gate, all with seed 42, whose reports `teleop_v12_stage create` binds
with the checkpoint into one gate file (schema 3).  The pipeline then
packages it:

```bash
uv run --locked --with onnxruntime --with "protobuf<7" python -m \
  mjlab_microban.scripts.export_teleop_v12_deployment \
  --checkpoint logs/rsl_rl/mjlab_microban_teleop_v12/<run>/model_<N>.pt \
  --stage-gate <state>/pico_judgment/<run>_model_<N>_gate.json \
  --output <state>/release/pico_teleop.onnx --force
```

`--force` only permits the final atomic rename to replace an existing file
after every check passed; on any failure the previous output is unchanged.
The packager, on CPU:

1. rebuilds and compares the gate with the current evaluator code, rehashing
   the checkpoint, the three reports and the gate ONNX; requires
   `status=pass`, the final profile, a completed-update count between
   `PICO_MIN_FINAL_UPDATES` and `PICO_TOTAL_UPDATES` and the checkpoint named
   after the gate's iteration;
2. captures the checkpoint bytes, revalidates the frozen walker and its probe
   (bootstrap provenance) and the corrected bilateral site order, and refuses
   dry-run evidence unless it packages a dry run;
3. exports a fixed-shape `obs[1,83] -> actions[1,18]` float32 graph;
4. writes the robot's metadata from the validated evidence (docs/policies.md):
   the policy contract, the servo gain, the HOME, the PICO schedule, the
   per-joint runtime guard derived from the final tracking envelope, the
   startup self-test observations of the final tracking rollouts, the target
   ranges and frames;
5. checks parity of PyTorch, ONNX `ReferenceEvaluator` and ONNX Runtime
   `CPUExecutionProvider` on a deterministic 64-sample corpus before and after
   the metadata is attached, and publishes with `os.replace` and a directory
   `fsync`.

It no longer runs or hashes the robot's sources: the robot checks a release
when it is installed (its `tools/validate_policies.py` on the manifest and its
tests) and at every start (the self-test observations).

## Tracking profile

One profile, `full_body_reachable_performance_perturbation_v2_deployed_accuracy_v1`,
the same at every HOME: all scenarios (low forward, both hand corners, both
keypoint corners, both feet, two mixed twists), the push perturbation, the
target-column ablation of hands and feet, and these limits:

| limit | value |
| --- | --- |
| hand RMS | 0.040 m |
| hand P95 | 0.07 m |
| foot RMS | 0.05 m |
| foot P95 | 0.08 m |

plus no falls, finite values, actual soft-limit overshoot <= 5 deg, raw-action
recurrence, forced HMD motion, observation coverage and the twist directional
response.  Hand RMS 0.040 m is the user's decision for every HOME.

Focused CPU-only tests:

```bash
uv run --locked --with pytest python -m pytest -q \
  tests/test_teleop_v12_deployment.py tests/test_teleop_v12_stage.py tests/test_pico_schedule.py
```
