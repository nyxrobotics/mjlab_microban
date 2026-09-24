#!/usr/bin/env bash
# Reproducible v7 latent-PPO canary with the neutral-preserving state guard.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly EXPECTED_SHA256="b0bcdadac39716be784207dd6b2b93157162a3e80650e23c05f490c400b9e141"
readonly VELOCITY_CHECKPOINT="${MICROBAN_TELEOP_BOOTSTRAP_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/xc330_velocity/model_14999.pt}"

if [[ ! -f "${VELOCITY_CHECKPOINT}" ]]; then
    echo "Audited velocity checkpoint not found: ${VELOCITY_CHECKPOINT}" >&2
    exit 2
fi
actual_sha256="$(sha256sum -- "${VELOCITY_CHECKPOINT}" | cut -d' ' -f1)"
if [[ "${actual_sha256}" != "${EXPECTED_SHA256}" ]]; then
    echo "Velocity checkpoint SHA-256 mismatch." >&2
    echo "Expected: ${EXPECTED_SHA256}" >&2
    echo "Actual:   ${actual_sha256}" >&2
    exit 2
fi

# 251 completed updates materialize model_250.pt (suffixes are zero-based).
# The three-seed gate runs this wrapper separately with seeds 42, 43 and 44.
export MICROBAN_TELEOP_TARGET_ITERS="${MICROBAN_TELEOP_TARGET_ITERS:-251}"
export MICROBAN_TELEOP_NUM_ENVS="${MICROBAN_TELEOP_NUM_ENVS:-4096}"
export MICROBAN_TELEOP_SAVE_INTERVAL="${MICROBAN_TELEOP_SAVE_INTERVAL:-50}"
exec "${SCRIPT_DIR}/train_microban_teleop.sh" train \
    --agent.bootstrap-velocity-checkpoint "${VELOCITY_CHECKPOINT}" \
    --agent.bootstrap-velocity-checkpoint-sha256 "${EXPECTED_SHA256}" \
    --agent.save-pristine-checkpoint True \
    "$@"
