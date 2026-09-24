#!/usr/bin/env bash
# Deliberately reject the retired v5 contract before any training starts.
set -euo pipefail

echo "Training contract v5 is rejected after the dynamic joint-limit failure; use train_microban_teleop_v8_stage.sh start." >&2
exit 2
