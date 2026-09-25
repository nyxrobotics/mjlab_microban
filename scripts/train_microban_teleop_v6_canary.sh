#!/usr/bin/env bash
# Deliberately reject the retired v6 contract before any training starts.
set -euo pipefail

echo "Training contract v6 is rejected after PPO action-space numerical failure; use scripts/train_microban_teleop_v9.sh after producing an accepted safe-velocity source." >&2
exit 2
