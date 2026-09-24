#!/usr/bin/env bash
# Historical entry point retained only to reject the superseded v7 recipe.
set -euo pipefail

echo "Training contract v7 is rejected; its bootstrap preflight is no longer valid." >&2
echo "Use scripts/run_microban_teleop_v8_preflight.sh." >&2
exit 2
