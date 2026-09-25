#!/usr/bin/env bash
# Canonical public name for the current v9 stage evaluator.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/evaluate_microban_teleop_v8_stage.sh" "$@"
