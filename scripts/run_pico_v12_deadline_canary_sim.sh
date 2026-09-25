#!/usr/bin/env bash
# Exact model10099 canonical-v12 policy in the local Microban simulation only.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly EXPECTED_CHECKPOINT_SHA256="86a81f45f34d91036ab138963c835a3e78db98224f523d3484e1d3ed335082ff"
readonly EXPECTED_RECEIPT_SHA256="e2ae7b6a59f1f5e641b98d316a8ada2b8d56d82aa6154db2e070771d117d2dad"

usage() {
    cat >&2 <<'EOF'
usage: run_pico_v12_deadline_canary_sim.sh CHECKPOINT POST_CANARY_PASS_RECEIPT RECEIPT_SHA256 [live options...]

Runs the exact accepted model10099 deadline canary against the canonical
Mjlab-Teleop-V12-Microban task. PICO input drives only a local simulated robot;
this path contains no robot address, motor sender, training, saving, or export.
The checkpoint and post-canary PASS receipt are authenticated before loading.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if (($# < 3)); then
    usage
    exit 2
fi

checkpoint=$1
receipt=$2
receipt_sha256=$3
shift 3

if [[ ! -f "${checkpoint}" || -L "${checkpoint}" ]]; then
    echo "[FAIL] checkpoint must be a regular non-symlink file: ${checkpoint}" >&2
    exit 2
fi
if [[ ! -f "${receipt}" || -L "${receipt}" ]]; then
    echo "[FAIL] receipt must be a regular non-symlink file: ${receipt}" >&2
    exit 2
fi
if [[ ! "${receipt_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "[FAIL] receipt SHA-256 must be 64 lowercase hexadecimal characters" >&2
    exit 2
fi
if [[ "${receipt_sha256}" != "${EXPECTED_RECEIPT_SHA256}" ]]; then
    echo "[FAIL] receipt is not the pinned post-canary PASS receipt" >&2
    exit 2
fi

checkpoint="$(realpath -- "${checkpoint}")"
receipt="$(realpath -- "${receipt}")"
if [[ "$(sha256sum -- "${checkpoint}" | cut -d ' ' -f 1)" != "${EXPECTED_CHECKPOINT_SHA256}" ]]; then
    echo "[FAIL] checkpoint is not the pinned model10099" >&2
    exit 2
fi
if [[ "$(sha256sum -- "${receipt}" | cut -d ' ' -f 1)" != "${receipt_sha256}" ]]; then
    echo "[FAIL] post-canary PASS receipt SHA-256 mismatch" >&2
    exit 2
fi

exec "${SCRIPT_DIR}/run_pico_legacy_fallback.sh" launch \
    --v12-deadline-canary-checkpoint "${checkpoint}" \
    --v12-deadline-canary-acceptance-receipt "${receipt}" \
    --v12-deadline-canary-acceptance-receipt-sha256 "${receipt_sha256}" \
    "$@"
