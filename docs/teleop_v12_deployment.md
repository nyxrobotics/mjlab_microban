# PICO package (`export_teleop_v12_deployment`)

The PICO policy of a release is the last checkpoint of its one training run,
`model_14999` (15000 updates, `PICO_TOTAL_UPDATES` in
`mjlab_microban/schedules.py`).  It is judged once
(`scripts/retrain_all_for_home.py`, step pico): the 9x300 locomotion
evaluation, the tracking evaluation under the one final profile and the ONNX
parity gate, all with seed 42, whose reports `teleop_v12_stage create` binds
with the checkpoint into one gate file (schema 3).  The pipeline then
packages it:

```bash
uv run --locked python -m \
  mjlab_microban.scripts.export_teleop_v12_deployment \
  --checkpoint logs/rsl_rl/mjlab_microban_teleop_v12/<run>/model_14999.pt \
  --stage-gate <state>/pico_judgment/<run>_model_14999_gate.json \
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
   for each.  It also reads the run logger's `git/mjlab_microban.diff`, refuses
   a run that started with tracked changes, and embeds the exact training
   commit, branch and SHA-256 of that Git record in the ONNX metadata;
5. checks parity of PyTorch, ONNX `ReferenceEvaluator` and ONNX Runtime
   `CPUExecutionProvider` on a deterministic 64-sample corpus before and after
   the metadata is attached, runs the robot's self-test rule on the final
   file, and publishes with `os.replace` and a directory `fsync`.

It does not run or hash the robot's sources: the robot checks a release when
it is installed (`tools/validate_policies.py src/agents` and its tests) and at
every start (the self-test).

The Git identity is the state captured when training started, not the branch
tip at packaging time.  In particular, documentation-only commits made while
a run is executing do not relabel the trained policy.

## Finalize a PICO run started outside the release pipeline

Normally `scripts/retrain_all_for_home.py` runs the three judgments, creates
the stage gate and packages the policy.  If the same production training was
started by hand, run the identical sequence below from this repository after
its final `model_14999.pt` exists.  The first two commands use the GPU; the
remaining commands are CPU-only.  Keep the checkpoint's `model_14999.pt`
name: the packager verifies it against iteration 14999.

```bash
set -euo pipefail

CKPT=logs/rsl_rl/mjlab_microban_teleop_v12/<run>/model_14999.pt
RUN=$(basename -- "$(dirname -- "$CKPT")")
OUT="artifacts/pico_release/$RUN"
PREFIX="$OUT/model_14999"
mkdir -p -- "$OUT"
test "$(basename -- "$CKPT")" = model_14999.pt
CHECKPOINT_SHA=$(sha256sum -- "$CKPT" | awk '{print $1}')

uv run --locked python -m mjlab_microban.scripts.evaluate_teleop_v12_checkpoint \
  "$CKPT" --expected-sha256 "$CHECKPOINT_SHA" --device cuda:0 \
  --seed 42 --steps 300 --settle-steps 50 \
  --output "${PREFIX}_9x300.json" --force

uv run --locked python -m mjlab_microban.scripts.evaluate_teleop_v12_tracking \
  "$CKPT" --expected-sha256 "$CHECKPOINT_SHA" --device cuda:0 \
  --seed 42 --output "${PREFIX}_tracking.json" --force

CUDA_VISIBLE_DEVICES='' uv run --locked python -m \
  mjlab_microban.scripts.teleop_v12_onnx_gate \
  "$CKPT" --expected-sha256 "$CHECKPOINT_SHA" \
  --onnx "${PREFIX}.onnx" --output "${PREFIX}_onnx.json" --force

CUDA_VISIBLE_DEVICES='' uv run --locked python -m \
  mjlab_microban.scripts.teleop_v12_stage create \
  "$CKPT" "${PREFIX}_9x300.json" "${PREFIX}_tracking.json" \
  "${PREFIX}_onnx.json" "${PREFIX}_gate.json" --force

CUDA_VISIBLE_DEVICES='' uv run --locked python -m \
  mjlab_microban.scripts.teleop_v12_stage validate \
  "${PREFIX}_gate.json" "$CKPT"

CUDA_VISIBLE_DEVICES='' uv run --locked python -m \
  mjlab_microban.scripts.export_teleop_v12_deployment \
  --checkpoint "$CKPT" --stage-gate "${PREFIX}_gate.json" \
  --output "$OUT/pico_teleop.onnx" --force
sha256sum -- "$CKPT" "$OUT/pico_teleop.onnx" "${PREFIX}"*.json
```

`set -e` is intentional: a judgment writes its failed report and exits 1, so
the gate must not be created after a failed judgment.  `--force` makes this
sequence safe to repeat at the same paths; every JSON and ONNX publication is
atomic, and the gate and packager re-hash all of their inputs.  Do not change
the canonical seed, step counts or tracking profile: `teleop_v12_stage`
rejects non-canonical evidence.  The parity ONNX (`${PREFIX}.onnx`) and its
JSON report are one evidence pair; if either is regenerated, run the ONNX
gate and the stage-gate creation again before packaging.

## PICO judgment

One profile, `pico_feet_push_still_arms_v1`, the same at every HOME
(`scripts/evaluate_teleop_v12_tracking.py`).  Each scenario runs 64
environments on seeds 42 and 43 with the HMD neck moving; the arms are driven
from outside as on the robot (HOME, raised forward 70 deg, one arm reaching
out, or moved as in training).

| check | scenarios | pass line |
| --- | --- | --- |
| feet (J1) | standing, no push: one foot up 20 / 40 mm and the four corners (+-24, +-24, 40) mm, left and right mirrored, two corners also with the arm on that side reaching out; both feet by the same (+8, +8, 16) / (-8, -8, 16) mm; reached through teleop's 0.12 m/s ramp, scored from 1 s after the change | one foot: height above the floor >= 0.7 x dz (median environment), support foot moves <= 10 mm; both feet (a crouch in the trunk frame): the trunk comes down >= 0.7 x dz; the one foot seen from the other within max(0.3 x their target difference, 8 mm) RMS |
| pushes (J2) | standing and the 9x300 commands, a 0.4 m/s kick every second from eight directions | falls at most 5 points more often than the walker the adapter was built on |
| standing still (J3) | standing with the HMD moving and the arms at HOME, raised, moving or one reaching out | <= 0.5 touchdowns per second and the standing drift of `twist_pass_line.py` |
| arms (J4) | the 9x300 commands with the arms raised or moving | v >= min(0.2 \|c\|, v_HOME) along the command (slower is accepted; v_HOME: the arms at HOME) |

plus no falls without pushes, finite values, actual soft-limit overshoot of
the twelve leg joints <= 0.25 rad, raw-action recurrence, forced HMD motion
and the arm-target observation.  There is no hand-tracking limit: the arms follow the robot's
pico_arms, not the policy.

Focused CPU-only tests:

```bash
uv run --locked --with pytest python -m pytest -q \
  tests/test_teleop_v12_deployment.py tests/test_teleop_v12_stage.py tests/test_pico_schedule.py
```
