#!/usr/bin/env bash
# Explicit simulation-only launcher for a marker-authenticated v12 preview.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat >&2 <<'EOF'
usage: run_pico_v12_preview.sh PREVIEW_CHECKPOINT FINAL_PASS_RECEIPT RECEIPT_SHA256 [live_pico_teleop_sim options...]

Runs a permanently non-deployable contract-v12 full-body preview with the real
PICO 4 Ultra and a simulated Microban. It reuses the isolated simulation pairing
(control port 63903 and camera port 8081) and has no robot or motor output.

The checkpoint must carry the exact preview_non_deployable/teleop_v12_preview
lineage marker. FINAL_PASS_RECEIPT must be the hash-bound strict or visual-only
full-body acceptance receipt for those exact checkpoint bytes. Missing, failed,
phase-1, legacy-v1, or altered artifacts are rejected before actor loading.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if [[ $# -lt 3 ]]; then
    usage
    exit 2
fi

preview_checkpoint=$1
acceptance_receipt=$2
receipt_sha256=$3
shift 3
if [[ ! -f "${preview_checkpoint}" || -L "${preview_checkpoint}" ]]; then
    echo "[FAIL] preview checkpoint must be a regular non-symlink file: ${preview_checkpoint}" >&2
    exit 2
fi
if [[ ! -f "${acceptance_receipt}" || -L "${acceptance_receipt}" ]]; then
    echo "[FAIL] final acceptance receipt must be a regular non-symlink file: ${acceptance_receipt}" >&2
    exit 2
fi
if [[ ! "${receipt_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "[FAIL] receipt SHA-256 must be 64 lowercase hexadecimal characters" >&2
    exit 2
fi
preview_checkpoint="$(realpath -- "${preview_checkpoint}")"
acceptance_receipt="$(realpath -- "${acceptance_receipt}")"
if [[ "$(sha256sum -- "${acceptance_receipt}" | cut -d ' ' -f 1)" != "${receipt_sha256}" ]]; then
    echo "[FAIL] final acceptance receipt SHA-256 mismatch" >&2
    exit 2
fi

exec "${SCRIPT_DIR}/run_pico_legacy_fallback.sh" launch \
    --v12-preview-checkpoint "${preview_checkpoint}" \
    --v12-preview-acceptance-receipt "${acceptance_receipt}" \
    --v12-preview-acceptance-receipt-sha256 "${receipt_sha256}" \
    "$@"
