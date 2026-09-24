#!/usr/bin/env bash
# Deliberately reject the retired v5 contract before any training starts.
set -euo pipefail

echo "Training contract v5 is rejected after the dynamic joint-limit failure; use run_microban_teleop_v7_preflight.sh." >&2
exit 2
