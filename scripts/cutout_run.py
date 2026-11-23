#!/usr/bin/env python3
"""Run the decision layer on a case directory.

    python scripts/cutout_run.py check-cache --case <case-dir> --cache-dir <cache>
    python scripts/cutout_run.py run         --case <case-dir> --cache-dir <cache> --out out

A case directory contains ``case.json`` and the source image. The capture cache
holds the three proposal products (P1/P2/P3) as described in SKILL.md. Both are
plain directories, so the same command works with a folder a human assembled by
hand and with one written by an automated capture step.

Exit codes: 0 = pass, 1 = uncertain (see scorecard.json), 2 = bad input or a
missing capture product.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from cutoutlib import cache, recipes  # noqa: E402
from cutoutlib.cache import CacheMiss  # noqa: E402
from cutoutlib.case import CaseError, load_case  # noqa: E402


def cmd_check_cache(args) -> int:
    case = load_case(args.case)
    print(f"case: {case.id}")
    print(f"  source: {case.source}")
    print(f"  keep: {case.keep}")
    print(f"  exclude: {case.exclude}")
    print(f"  thin: {case.thin}")
    try:
        prop = cache.load(case.id, args.tag, root=args.cache_dir)
    except CacheMiss as exc:
        print(f"cache: MISSING -> {exc}")
        return 2
    shape = prop.bria.shape
    print(f"cache: {prop.root}")
    print(f"  P1 bria.png   alpha {shape[1]}x{shape[0]}, "
          f"foreground {int((prop.bria > 127).sum())} px")
    print(f"  P3 key.png    rgb   {prop.key_rgb.shape[1]}x{prop.key_rgb.shape[0]}")
    for layer in prop.layers:
        frame = (layer.bbox[0] >= 0 and layer.bbox[1] >= 0
                 and layer.bbox[2] <= shape[1] and layer.bbox[3] <= shape[0])
        print(f"  P2 {layer.name:<12} kind={layer.label!r:<28} "
              f"bbox={list(layer.bbox)} {'ok' if frame else 'OUT OF FRAME'}")
    bad = [l.name for l in prop.layers
           if not (0 <= l.bbox[0] < l.bbox[2] <= shape[1] and 0 <= l.bbox[1] < l.bbox[3] <= shape[0])]
    if bad:
        print("cache: bbox must be in source-image coordinates; out of frame: " + ", ".join(bad))
        return 2
    print("cache: ok (P1 + P2 + P3 all present)")
    return 0


def cmd_run(args) -> int:
    from cutoutlib.pipeline import run_case

    case = load_case(args.case)
    recipe = recipes.make("auto")
    if args.recipe and args.recipe != "auto":
        recipe = recipes.load(args.recipe)
    thresholds = None
    if args.thresholds:
        thresholds = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    res = run_case(case, recipe, out_dir=args.out, write=True,
                   thresholds=thresholds, cache_root=args.cache_dir)
    card = res["card"]
    m = card["metrics"]
    print(f"[{case.id}] status={card['status']} score={card['score']} "
          f"blocked_layer={card.get('blocked_layer')}")
    for key in ("halo", "stray", "hole", "coverage", "continuity", "pixel_fidelity"):
        print(f"  {key:<15} = {m.get(key)}")
    if card["failures"]:
        for f in card["failures"]:
            print(f"  failed: {f['metric']} = {f['value']} (limit {f['limit']})")
    print(f"deliverables: {args.out}/{case.id}.{{rgba.png,mask.png,scorecard.json,recipe.json,proof.png}}")
    return 0 if card["status"] == "pass" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check-cache", help="validate the three proposal products without running")
    p.add_argument("--case", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--tag", default=None)
    p.set_defaults(func=cmd_check_cache)

    p = sub.add_parser("run", help="run the decision layer and write deliverables")
    p.add_argument("--case", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--out", default="out")
    p.add_argument("--recipe", default="auto")
    p.add_argument("--thresholds", default=None, help="optional JSON overrides (rarely needed)")
    p.set_defaults(func=cmd_run)

    args = parser.parse_args()
    try:
        return args.func(args)
    except CacheMiss as exc:
        print(f"capture cache missing: {exc}")
        print("expected next to each other: bria.png (P1), key.png (P3), layers/*.png (P2), meta.json")
        return 2
    except CaseError as exc:
        print(f"case directory problem: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
