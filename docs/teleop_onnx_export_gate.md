# Teleop ONNX export gate

Use the checked-in exporter for every Microban PICO policy that may be copied to
the robot. A deployable export requires the exact schema-3 final gate receipt:

```bash
uv run python -m mjlab_microban.scripts.export_teleop_onnx \
  --checkpoint logs/rsl_rl/mjlab_microban_teleop/<run>/model_19999.pt \
  --acceptance-receipt \
    artifacts/teleop_v8_gates/<run>_boundary_20000_gate.json \
  --require-final-acceptance \
  --device cpu \
  --output artifacts/microban_teleop.onnx
```

Omitting the receipt is permitted for training-time and diagnostic exports, but
the resulting ONNX is explicitly marked `deployment_accepted=false` and must be
rejected by the robot runtime. `--require-final-acceptance` prevents an operator
from accidentally publishing such a diagnostic artifact as a deployment build.

Keep the checkpoint's `model_<iteration>.pt` filename. The iteration is parsed
from that name and recorded in the ONNX together with the checkpoint SHA-256,
completed-update count, exporter source SHA-256, source Git commit, and dirty
state. The exporter also revalidates the checkpoint's deterministic training
manifest and copies its complete-manifest SHA-256, exact source-tree SHA-256,
recipe, actor initialization, canonical/generic mode, and stage/parent lineage.
A generic checkpoint is always represented as `canonical_training_stage=false`
with `none` lineage values. The exporter hashes the checkpoint again immediately
before publication; if it changed while being loaded or exported, publication
fails.

For a deployable artifact, the exporter additionally requires canonical stage
`18000->20000` and revalidates the complete final receipt. The receipt must be
an exact schema-3 `pass` for the same checkpoint SHA-256 and training provenance,
the current evaluator and acceptance revisions, and the current evaluator source
SHA-256. Its three nominal and three moving-HMD report files must still match
their recorded SHA-256s, seeds, 1,000-step/50-settle configuration, canonical
scenario order, safety/acceptance results, and moving-HMD excursion evidence.
The ONNX records the receipt SHA-256, so it binds the exact six-report evidence
rather than merely claiming that a final-stage checkpoint exists.

The export gate runs deterministic PyTorch and ONNX inference over 16 fixed
83-value observations. The corpus contains a level neutral pose, lower and upper
finite boundaries, a midpoint, and seeded (`20260924`) schema-aware samples.
It compares the single 18-value action output using `atol=1e-5` and `rtol=1e-4`
through ONNX's checked-in `ReferenceEvaluator`, so `onnxruntime` is not needed on
the training machine. Shape drift, non-finite outputs, a numerical mismatch,
ambiguous metadata, or a changed checkpoint all fail the export.

Metadata is attached before the final ONNX validation and numerical comparison.
Only a metadata-complete, parity-verified temporary file replaces the requested
output, using a same-directory atomic rename. A failure therefore preserves the
last known-good output. Automatic checkpoint ONNX exports use the same gate; an
automatic export failure is logged without stopping PPO training.

The robot policy loader must fail closed on these deployment fields. It requires
`canonical_training_stage=true`, `training_provenance_mode=canonical_v8_stage`,
stage `18000->20000`, `deployment_accepted=true`, receipt schema `3`, status
`pass`, and boundary `20000`, in addition to validating all SHA/revision/recipe
and actor-initialization fields. To inspect them without running the robot:

```bash
uv run python - <<'PY'
import onnx

model = onnx.load("artifacts/microban_teleop.onnx")
for item in model.metadata_props:
    if item.key.startswith(
        ("checkpoint_", "training_", "acceptance_", "exporter_", "onnx_parity_")
    ) or item.key in ("canonical_training_stage", "deployment_accepted"):
        print(f"{item.key}={item.value}")
PY
```

The training identity fields are:

- `training_provenance_schema_version`, `training_provenance_sha256`, and
  `training_source_tree_sha256`;
- `training_recipe_revision`, `training_actor_initialization`, and
  `training_provenance_mode`;
- `canonical_training_stage`, `training_stage_start_boundary`,
  `training_stage_target_boundary`, `training_parent_checkpoint_sha256`, and
  `training_parent_gate_sha256`.

The acceptance identity fields are `deployment_accepted`,
`acceptance_receipt_schema_version`, `acceptance_receipt_sha256`,
`acceptance_status`, `acceptance_boundary`, `acceptance_evaluator_revision`,
`acceptance_revision`, `acceptance_evaluator_source_sha256`,
`acceptance_checkpoint_sha256`, `acceptance_training_provenance_sha256`,
`acceptance_recipe_revision`, and the nominal and moving-HMD report counts. The
three repeated acceptance identities must exactly equal the checkpoint,
training-provenance and recipe fields outside the receipt namespace. A
nonaccepted artifact uses literal `none` for every unavailable acceptance
identity and `0` for both counts; missing keys are never equivalent to a
diagnostic export.

Run the focused regression tests after changing the exporter or policy wrapper:

```bash
uv run python -m unittest tests.test_teleop_onnx_parity_gate -v
uv run python -m unittest tests.test_microban_policy_export -v
```
