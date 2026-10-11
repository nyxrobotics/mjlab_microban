"""Build the whole-body training and evaluation sets from GMR's retargeted PKL files.

  uv run --locked python src/mjlab_microban/scripts/wholebody_build_dataset.py \\
      [--retarget-dir data/wholebody/lafan1/retarget] [--out-dir data/wholebody/lafan1]

Writes ``<out-dir>/train/motion.h5``, ``<out-dir>/eval/motion.h5`` and
``<out-dir>/report.json`` (per motion kind: the files of each set and the
share of frames cut, by reason), and prints the report.  LAFAN1 is CC BY-NC-ND
4.0: everything under data/wholebody/ stays out of git and off the network.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from mjlab_microban.wholebody.data.build import build_dataset, format_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retarget-dir", type=Path, default=Path("data/wholebody/lafan1/retarget"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/wholebody/lafan1"))
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = parser.parse_args()
    report = build_dataset(args.retarget_dir, args.out_dir, workers=args.workers)
    print(format_report(report))


if __name__ == "__main__":
    main()
