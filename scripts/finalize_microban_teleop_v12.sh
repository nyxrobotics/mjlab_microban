#!/usr/bin/env bash
# Evaluate, package, runtime-validate, and receipt one final v12 checkpoint.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly LOG_ROOT="${PROJECT_ROOT}/logs/rsl_rl/mjlab_microban_teleop_v12"
readonly GATE_ROOT="${PROJECT_ROOT}/artifacts/teleop_v12_gates"
readonly RELEASE_ROOT="${PROJECT_ROOT}/artifacts/teleop_v12_releases"

usage() {
    cat <<'EOF'
Usage:
  scripts/finalize_microban_teleop_v12.sh RUN_NAME [OUTPUT_ONNX] [--force] [--reuse-gate]

Runs the authoritative final 9x300 locomotion, full tracking/safety, ONNX
parity, stage-gate, deployment packaging, and real Microban CPU-runtime checks
for RUN_NAME/model_14999.pt.  The default output is a run-specific ONNX under
artifacts/teleop_v12_releases.  --force permits replacement of both the ONNX
and its deployment receipt; it never weakens any acceptance check.
--reuse-gate resumes packaging from an already-created gate after validating
that gate against the exact final checkpoint and hash-bound reports.
EOF
}

fail() { echo "$*" >&2; exit 2; }

(( $# >= 1 && $# <= 4 )) || { usage >&2; exit 2; }
[[ "$1" != "-h" && "$1" != "--help" ]] || { usage; exit 0; }

run_name="$1"
shift
[[ "${run_name}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] \
    || fail "RUN_NAME must be one safe literal directory name."

output="${RELEASE_ROOT}/${run_name}_model_14999.onnx"
force=0
reuse_gate=0
if (( $# > 0 )) && [[ "$1" != --* ]]; then
    output="$1"
    shift
fi
while (( $# > 0 )); do
    case "$1" in
        --force) force=1 ;;
        --reuse-gate) reuse_gate=1 ;;
        *) fail "Unknown argument: $1" ;;
    esac
    shift
done

readonly checkpoint="${LOG_ROOT}/${run_name}/model_14999.pt"
readonly gate="${GATE_ROOT}/${run_name}_model_14999_gate.json"
readonly receipt="${RELEASE_ROOT}/${run_name}_model_14999_deployment_receipt.json"
output="$(realpath -m -- "${output}")"
[[ "${output}" != "${receipt}" ]] \
    || fail "Deployment ONNX and receipt must be different paths."

[[ -f "${checkpoint}" && ! -L "${checkpoint}" ]] \
    || fail "Final checkpoint is not ready: ${checkpoint}"
if (( force == 0 )); then
    [[ ! -e "${output}" ]] || fail "Deployment ONNX exists (pass --force): ${output}"
    [[ ! -e "${receipt}" ]] || fail "Deployment receipt exists (pass --force): ${receipt}"
fi

mkdir -p -- "$(dirname -- "${output}")" "${RELEASE_ROOT}"
temporary_receipt="$(mktemp --tmpdir="${RELEASE_ROOT}" \
    ".${run_name}_model_14999_receipt.XXXXXX")"
cleanup() { rm -f -- "${temporary_receipt}"; }
trap cleanup EXIT

cd -- "${PROJECT_ROOT}"
if (( reuse_gate == 0 )); then
    scripts/evaluate_microban_teleop_v12_stage.sh "${run_name}" 14999
else
    [[ -f "${gate}" && ! -L "${gate}" ]] \
        || fail "Reusable final gate is not ready: ${gate}"
fi

# Validate the just-written manifest independently before packaging consumes it.
uv run --locked python -m mjlab_microban.scripts.teleop_v12_stage \
    validate "${gate}" "${checkpoint}" >/dev/null

export_args=("${run_name}" "${output}")
(( force == 0 )) || export_args+=(--force)
scripts/export_microban_teleop_v12_deployment.sh "${export_args[@]}" \
    >"${temporary_receipt}"

# The packager already performs the real runtime smoke.  Bind its report to the
# exact bytes published above, then atomically publish the human/audit receipt.
uv run --locked python - "${temporary_receipt}" "${checkpoint}" "${gate}" \
    "${output}" "${receipt}" <<'PY'
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


temporary, checkpoint, gate, output, receipt = map(
    lambda value: Path(value).resolve(), sys.argv[1:]
)
report = json.loads(temporary.read_text(encoding="utf-8"))
parity = report.get("parity")
runtime = report.get("microban_runtime_validator")
smoke = runtime.get("onnxruntime_compatibility_smoke") if isinstance(runtime, dict) else None
required = (
    report.get("schema_version") == 1,
    report.get("status") == "pass",
    report.get("completed_updates") == 15_000,
    report.get("output") == str(output),
    report.get("output_sha256") == sha256(output),
    report.get("checkpoint_sha256") == sha256(checkpoint),
    report.get("stage_gate_sha256") == sha256(gate),
    isinstance(parity, dict),
    isinstance(runtime, dict) and runtime.get("status") == "pass",
    isinstance(smoke, dict) and smoke.get("status") == "pass",
    isinstance(smoke, dict) and smoke.get("providers") == ["CPUExecutionProvider"],
)
if not all(required):
    raise SystemExit("Deployment report is not a complete hash-bound runtime pass")
for name in (
    "reference_maximum_absolute_error",
    "onnxruntime_cpu_maximum_absolute_error",
):
    value = parity.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SystemExit(f"Deployment parity value is invalid: {name}")

with temporary.open("rb") as stream:
    os.fsync(stream.fileno())
os.replace(temporary, receipt)
directory_fd = os.open(receipt.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
PY

trap - EXIT
echo "[PASS] final checkpoint: ${checkpoint}"
echo "[PASS] hash-bound stage manifest: ${gate}"
echo "[PASS] runtime-validated deployment ONNX: ${output}"
echo "[PASS] deployment receipt: ${receipt}"
