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
3. exports a fixed-shape `obs[1,83] -> actions[1,18]` float32 graph;
4. writes the robot's contract microban-policy-1 from the validated evidence
   (docs/policies.md): the HOME stamp and layout, the PICO targets, frame and
   curriculum, the per-joint raw-action guard derived from the final tracking
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

## PICO judgment

One profile, `pico_feet_push_still_arms_v1`, the same at every HOME
(`scripts/evaluate_teleop_v12_tracking.py`).  Each scenario runs 64
environments on seeds 42 and 43 with the HMD neck moving; the arms are driven
from outside as on the robot (HOME, raised forward 70 deg, or moved as in
training).

| check | scenarios | pass line |
| --- | --- | --- |
| feet (J1) | standing, no push: one foot up 20 / 40 mm, the four corners (+-24, +-24, 40) mm, both feet (+-8, +-8, 16) mm; left and right mirrored; reached through teleop's 0.12 m/s ramp, scored from 1 s after the change | height above the floor >= 0.7 x target (median environment); the lifted foot seen from the support foot within max(0.3 x target, 8 mm) RMS; support foot moves <= 10 mm; mirrored halves within 5 mm |
| pushes (J2) | standing and the 9x300 commands, a 0.4 m/s kick every second from eight directions | falls at most 5 points more often than the walker the adapter was built on |
| standing still (J3) | standing with the HMD and the arms moving | <= 0.5 touchdowns per second and the standing drift of `twist_pass_line.py` |
| arms (J4) | the 9x300 commands with the arms raised or moving | speed along the command within 10 % of the speed with the arms at HOME |

plus no falls without pushes, finite values, actual soft-limit overshoot
<= 0.25 rad, raw-action recurrence, forced HMD motion and the arm-target
observation.  There is no hand-tracking limit: the arms follow the robot's
pico_arms, not the policy.

Focused CPU-only tests:

```bash
uv run --locked --with pytest python -m pytest -q \
  tests/test_teleop_v12_deployment.py tests/test_teleop_v12_stage.py tests/test_pico_schedule.py
```
