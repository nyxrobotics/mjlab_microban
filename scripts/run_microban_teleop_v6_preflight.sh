#!/usr/bin/env bash
# Deliberately reject the retired v6 contract before any training starts.
set -euo pipefail

echo "Training contract v6 is rejected after PPO action-space numerical failure; use run_microban_teleop_v8_preflight.sh." >&2
exit 2
