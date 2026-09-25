#!/usr/bin/env bash
# Explicit simulation-only launcher for a marker-authenticated v12 preview.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat >&2 <<'EOF'
usage: run_pico_v12_preview.sh PREVIEW_CHECKPOINT [live_pico_teleop_sim options...]

Runs a permanently non-deployable contract-v12 full-body preview with the real
PICO 4 Ultra and a simulated Microban. It reuses the isolated simulation pairing
(control port 63903 and camera port 8081) and has no robot or motor output.

The checkpoint must carry the exact preview_non_deployable/teleop_v12_preview
lineage marker. A missing, ordinary, or altered checkpoint is not loaded; the
same process retains the audited legacy model_14999 joystick fallback.
EOF
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    [[ $# -ge 1 ]] && exit 0
    exit 2
fi

preview_checkpoint=$1
shift
if [[ ! -f "${preview_checkpoint}" || -L "${preview_checkpoint}" ]]; then
    echo "[FAIL] preview checkpoint must be a regular non-symlink file: ${preview_checkpoint}" >&2
    exit 2
fi
preview_checkpoint="$(realpath -- "${preview_checkpoint}")"

exec "${SCRIPT_DIR}/run_pico_legacy_fallback.sh" launch \
    --v12-preview-checkpoint "${preview_checkpoint}" \
    "$@"
