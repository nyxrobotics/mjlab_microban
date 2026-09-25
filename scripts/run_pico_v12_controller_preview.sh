#!/usr/bin/env bash
# Simulation-only PICO controller-hand preview with exact-zero foot targets.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
    cat >&2 <<'EOF'
usage: run_pico_v12_controller_preview.sh PHASE1_CHECKPOINT PHASE1_PASS_RECEIPT PHASE1_RECEIPT_SHA256 DIRECT_OVERLAY_ACCEPTANCE_RECEIPT DIRECT_OVERLAY_RECEIPT_SHA256 [live_pico_teleop_sim options...]

Runs the audited legacy walking actor in simulated Microban and overlays only
the six arm joints from the PICO controllers' bounded, slew-limited IK targets.
Foot targets are forced to exact zero and Motion Tracker body targets are never
consumed. The phase-1 checkpoint and strict/visual PASS receipt are hash-bound and
strict-loaded. The separate direct-overlay receipt must reproduce a passing
45-case simulation adjudication before launch. This path remains permanently
simulation-only.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if [[ $# -lt 5 ]]; then
    usage
    exit 2
fi

checkpoint=$1
receipt=$2
receipt_sha256=$3
overlay_receipt=$4
overlay_receipt_sha256=$5
shift 5

if [[ ! -f "${checkpoint}" || -L "${checkpoint}" ]]; then
    echo "[FAIL] phase-1 checkpoint must be a regular non-symlink file: ${checkpoint}" >&2
    exit 2
fi
if [[ ! -f "${receipt}" || -L "${receipt}" ]]; then
    echo "[FAIL] phase-1 receipt must be a regular non-symlink file: ${receipt}" >&2
    exit 2
fi
if [[ ! "${receipt_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "[FAIL] phase-1 receipt SHA-256 must be 64 lowercase hexadecimal characters" >&2
    exit 2
fi
if [[ ! -f "${overlay_receipt}" || -L "${overlay_receipt}" ]]; then
    echo "[FAIL] direct-overlay receipt must be a regular non-symlink file: ${overlay_receipt}" >&2
    exit 2
fi
if [[ ! "${overlay_receipt_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "[FAIL] direct-overlay receipt SHA-256 must be 64 lowercase hexadecimal characters" >&2
    exit 2
fi

checkpoint="$(realpath -- "${checkpoint}")"
receipt="$(realpath -- "${receipt}")"
overlay_receipt="$(realpath -- "${overlay_receipt}")"
if [[ "$(sha256sum -- "${receipt}" | cut -d ' ' -f 1)" != "${receipt_sha256}" ]]; then
    echo "[FAIL] phase-1 receipt SHA-256 mismatch" >&2
    exit 2
fi
if ! (
    cd -- "${PROJECT_ROOT}"
    uv run --locked python "${PROJECT_ROOT}/scripts/adjudicate_direct_ik_overlay.py" \
        --verify-receipt "${overlay_receipt}" "${overlay_receipt_sha256}"
); then
    echo "[FAIL] direct-overlay simulation acceptance receipt did not verify" >&2
    exit 2
fi

exec "${SCRIPT_DIR}/run_pico_legacy_fallback.sh" launch \
    --v12-controller-preview-checkpoint "${checkpoint}" \
    --v12-controller-preview-acceptance-receipt "${receipt}" \
    --v12-controller-preview-acceptance-receipt-sha256 "${receipt_sha256}" \
    "$@"
