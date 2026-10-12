"""Make the whole-body data from the downloads (docs/wholebody_data.md).  CPU only.

  uv run --locked python src/mjlab_microban/scripts/wholebody_make_data.py all [--workers 12]

``all`` downloads LAFAN1, CMU and G1's model (``fetch``), retargets every
clip to Microban (``retarget``), cuts the long arm-branch flips
(``finalize``), builds the training and evaluation shards (``build``) and
links them into ``data/wholebody/mix`` (``mix``).  The PICO evaluation set
(``pico_eval``) and AMASS (``amass``) are included when their files are
under ``data/wholebody/raw`` (placed by hand).  ``--support`` retargets with
the support changes (so that Microban's zero-moment point stays in its
support, the contact hull shrunk by ``--support-margin-mm``, the feet held
where they stand; in this order, each only where the ones before leave the
zero-moment point outside: the trunk moved, tilted, the feet rolled and
placed elsewhere, at most ``--support-max-lateral-mm``,
``--support-max-forward-mm``, ``--support-max-tilt-deg``,
``--support-max-ankle-roll-deg``, ``--support-max-foot-mm``;
retarget/support.py); without it the targets are not changed.  Each step can
run alone:

  ... wholebody_make_data.py fetch
  ... wholebody_make_data.py retarget --sets cmu [--only names.txt]
  ... wholebody_make_data.py finalize --sets lafan1 cmu
  ... wholebody_make_data.py build --sets lafan1 cmu pico_eval
  ... wholebody_make_data.py mix --sets lafan1 cmu

and the tools for one clip or one set:

  ... wholebody_make_data.py file data/wholebody/raw/cmu/data/007/07_01.bvh --format bvh_cmu --out-dir /tmp/out
  ... wholebody_make_data.py render /tmp/out/07_01.npz
  ... wholebody_make_data.py quality data/wholebody/gmr/cmu data/wholebody/cmu/quality.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mjlab_microban.wholebody.retarget import pipeline
from mjlab_microban.wholebody.retarget.support import SupportSpec


def _present_sets(root: Path) -> list[str]:
    raw = root / "raw"
    sets = ["lafan1", "cmu"]
    if (raw / "amass" / "smplx_g").is_dir():
        sets.append("amass")
    if (raw / "pico").is_dir():
        sets.append("pico_eval")
    return sets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["all", "fetch", "retarget", "finalize", "build", "mix", "file", "render", "quality"])
    parser.add_argument("paths", nargs="*", type=Path, help="file: source files; render: npz files; quality: folder and output json")
    parser.add_argument("--root", type=Path, default=pipeline.ROOT)
    parser.add_argument("--sets", nargs="+", help="Default: lafan1 and cmu, and amass and pico_eval when their raw files exist")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--only", type=Path, help="retarget: a file with the clip names to redo")
    parser.add_argument("--format", default="bvh_cmu", help="file: the source format (skeleton.FORMATS)")
    parser.add_argument("--out-dir", type=Path, default=Path("."), help="file: the output folder")
    parser.add_argument("--support", action="store_true",
                        help="all, retarget, file: change the targets so that the zero-moment point stays in the support")
    parser.add_argument("--support-margin-mm", type=float,
                        help=f"with --support: the support is the contact hull shrunk inward by this "
                             f"(default {1000 * SupportSpec.margin_m:g})")
    # The changes' caps, in the order they are allowed (support.STAGES): (option, SupportSpec field, unit scale).
    caps = (("lateral-mm", "max_lateral_m", 1000.0, "the trunk's sideways shift"),
            ("forward-mm", "max_forward_m", 1000.0, "the trunk's forward shift"),
            ("tilt-deg", "max_tilt_deg", 1.0, "the trunk's tilt (roll and pitch each)"),
            ("ankle-roll-deg", "max_ankle_roll_deg", 1.0, "a foot's roll"),
            ("foot-mm", "max_foot_m", 1000.0, "a foot put elsewhere on the floor"))
    for option, field, unit, what in caps:
        parser.add_argument(f"--support-max-{option}", type=float,
                            help=f"with --support: at most this for {what} (default {unit * getattr(SupportSpec, field):g}; 0: not used)")
    args = parser.parse_args()
    support_spec = None
    given = {field: getattr(args, f"support_max_{option.replace('-', '_')}") / unit for option, field, unit, _ in caps
             if getattr(args, f"support_max_{option.replace('-', '_')}") is not None}
    if (args.support_margin_mm is not None or given) and not args.support:
        parser.error("--support-margin-mm and --support-max-* need --support")
    if args.support:
        margin = SupportSpec.margin_m if args.support_margin_mm is None else args.support_margin_mm / 1000
        try:
            support_spec = SupportSpec(margin_m=margin, **given)
        except ValueError as error:
            parser.error(str(error))

    if args.step == "file":
        from mjlab_microban.wholebody.retarget import core

        for path in args.paths:
            report = core.run(path, args.format, pipeline.FPS, args.out_dir,
                              smplx_model_dir=args.root / "raw" / "body_models" / "smplx", support_spec=support_spec)
            print(json.dumps(report, ensure_ascii=False))
        return
    if args.step == "render":
        from mjlab_microban.wholebody.retarget import render

        for path in args.paths:
            print(render.render(path, 640, 480, None))
        return
    if args.step == "quality":
        source, output = args.paths
        result = pipeline.quality(source, args.workers)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=1))
        print(json.dumps(result["summary"], ensure_ascii=False, indent=1))
        return

    root = args.root
    if args.step in ("all", "fetch"):
        pipeline.fetch(root)
    sets = args.sets or _present_sets(root)
    training = [name for name in sets if name in pipeline.TRAINING_SETS]
    if args.step in ("all", "retarget"):
        only = set(args.only.read_text().split()) if args.only else None
        if only and (unknown := sorted(only - set().union(*(pipeline.clip_names(n, root) for n in training)))):
            parser.error(f"--only names no clip of {', '.join(training)}: {unknown[:10]}")
        for name in training:
            try:
                errors = pipeline.retarget(name, root, args.workers, only, support_spec)
            except ValueError as error:  # --only with other support settings than the set's
                parser.error(str(error))
            if errors:
                print(f"{name}: {len(errors)} clips failed, see {root / 'gmr' / name / 'errors.json'}")
        if "pico_eval" in sets:
            pipeline.retarget_pico(root, min(args.workers, 8), support_spec)
    if args.step in ("all", "finalize"):
        for name in sets:
            print(json.dumps(pipeline.finalize(name, root)))
    if args.step in ("all", "build"):
        from mjlab_microban.wholebody.data.build import format_report

        for name in sets:
            print(f"== {name}\n{format_report(pipeline.build(name, root, args.workers))}")
    if args.step in ("all", "mix"):
        pipeline.mix(training, root)
        print(f"mix: {', '.join(training)} -> {root / 'mix'}/{{train,eval}}")


if __name__ == "__main__":
    main()
