#!/usr/bin/env bash
# Historical entry point retained only to reject v7 artifacts as current gates.
set -euo pipefail

echo "Training contract v7 is rejected and its reports cannot unlock v9 stages." >&2
echo "Use scripts/evaluate_microban_teleop_v9_stage.sh RUN_NAME." >&2
exit 2
