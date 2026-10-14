"""``python -m mjlab_microban.live``: the live retargeting runs without the training stack (mjlab, torch)."""

import os

os.environ.setdefault("MICROBAN_WITHOUT_TRAINING", "1")

from mjlab_microban.live.app import main

main()
