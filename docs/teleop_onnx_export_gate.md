# Teleop ONNX export gate

Use the checked-in exporter for every Microban PICO policy that may be copied to
the robot:

```bash
uv run python -m mjlab_microban.scripts.export_teleop_onnx \
  --checkpoint logs/rsl_rl/mjlab_microban_teleop/<run>/model_<iteration>.pt \
  --device cpu \
  --output artifacts/microban_teleop.onnx
```

Keep the checkpoint's `model_<iteration>.pt` filename. The iteration is parsed
from that name and recorded in the ONNX together with the checkpoint SHA-256,
completed-update count, exporter source SHA-256, source Git commit, and dirty
state. The exporter hashes the checkpoint again immediately before publication;
if it changed while being loaded or exported, publication fails.

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

The robot policy loader treats ONNX metadata as an extensible map: these
additional provenance keys do not change the existing required deployment
contract. To inspect them without running the robot:

```bash
uv run python - <<'PY'
import onnx

model = onnx.load("artifacts/microban_teleop.onnx")
for item in model.metadata_props:
    if item.key.startswith(("checkpoint_", "exporter_", "onnx_parity_")):
        print(f"{item.key}={item.value}")
PY
```

Run the focused regression tests after changing the exporter or policy wrapper:

```bash
uv run python -m unittest tests.test_teleop_onnx_parity_gate -v
uv run python -m unittest tests.test_microban_policy_export -v
```
