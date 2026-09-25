# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Canonical public entry point for contract-v9 stage validation."""

from mjlab_microban.scripts.teleop_v8_stage import (
    StageInterval,
    main,
    resolve_stage_interval,
    validate_interrupted_stage_checkpoint,
)

__all__ = (
    "StageInterval",
    "resolve_stage_interval",
    "validate_interrupted_stage_checkpoint",
)

if __name__ == "__main__":
    main()
