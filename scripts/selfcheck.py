#!/usr/bin/env python3
"""One-command self-check for this skill.

    python scripts/selfcheck.py

What it does, in order -- all local, no network, no credentials:

1. Verifies the required third-party libraries are importable; if not, says what
   to install instead of dumping an ImportError traceback.
2. Verifies the thresholds in ``cutoutlib/score.py`` still match
   ``docs/thresholds.md`` row by row (12 numeric rows + the p1_leak rule), and
   that the attribution table and the three uncertain rules are present.
3. Runs the offline demo (one source image + three hand-authored proposal files)
   through the full decision layer.
4. Prints the six primary metrics and the pass/uncertain verdict.

Output is ASCII so it stays readable in any console codepage.

Exit codes: 0 = demo passes, 1 = demo ran but is uncertain, 2 = environment or
package problem (missing library, missing demo file, threshold drift).
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIX = ("halo", "stray", "hole", "coverage", "continuity", "pixel_fidelity")


def _dependency_hint() -> int:
    try:
        import numpy  # noqa: F401
        import cv2  # noqa: F401
    except ImportError as exc:
        print("MISSING DEPENDENCY: " + str(exc.name))
        print("This package needs numpy and opencv-python only:")
        print("  pip install -r requirements.txt")
        print("  # or: pip install numpy opencv-python")
        print("(scipy / Pillow are not runtime dependencies of this package.)")
        return 2
    return 0


def _rows(markdown: str, header: str) -> list[list[str]]:
    """Return table rows under a '| a | b |' header line, cells stripped."""
    out: list[list[str]] = []
    started = False
    for line in markdown.splitlines():
        s = line.strip()
        if not started:
            if s.startswith("|") and header in s:
                started = True
            continue
        if not s.startswith("|"):
            if out:
                break
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):
            continue
        out.append(cells)
    return out


def _nums(cell: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", cell)]


def check_thresholds(score_mod) -> list[str]:
    """Compare docs/thresholds.md with DEFAULT_THRESHOLDS. Returns problem strings."""
    th = score_mod.DEFAULT_THRESHOLDS
    md = (ROOT / "docs" / "thresholds.md").read_text(encoding="utf-8")
    rows = _rows(md, "| 指标 | 阈值 |")
    expected = {
        "halo": ("halo_max", "le"),
        "stray": ("stray_max", "le"),
        "hole": ("hole_max", "le"),
        "coverage": ("coverage_min", "ge"),
        "continuity": ("continuity_min", "ge"),
        "pixel_fidelity": ("fidelity_max", "le"),
        "iou_ref": ("iou_ref_min", "ge"),
        "ref_cut": ("ref_cut_max", "le"),
        "exclude_hit": ("exclude_hit_max", "le"),
        "thin_cov": ("thin_cov_min", "ge"),
        "alpha_area_frac": ("alpha_area_frac_max", "le"),
        "alpha_evidence_ratio": ("alpha_evidence_ratio_max", "le"),
    }
    # The attribution table intentionally covers 12 rows: ref_cut is attributed
    # to "整体形状" through iou_ref, so it has no separate row in the source doc.
    expected_attr = {
        "halo", "stray", "hole", "coverage", "continuity", "pixel_fidelity",
        "iou_ref", "exclude_hit", "alpha_area_frac", "alpha_evidence_ratio",
        "p1_leak", "thin_cov",
    }
    problems: list[str] = []
    seen = set()
    numeric = 0
    for cells in rows:
        if len(cells) < 2:
            continue
        metric, limit = cells[0], cells[1]
        if metric == "p1_leak":
            values = _nums(limit.replace("P1", ""))
            if len(values) < 2:
                problems.append(f"p1_leak rule not parsed: {limit!r}")
                continue
            if (abs(values[0] - th["p1_area_frac_warn"]) > 1e-9
                    or abs(values[1] - th["p1_leak_ratio_max"]) > 1e-9):
                problems.append(
                    f"p1_leak drift: docs={values[:2]} code="
                    f"[{th['p1_area_frac_warn']}, {th['p1_leak_ratio_max']}]")
            seen.add("p1_leak")
            continue
        if metric not in expected:
            continue
        key, direction = expected[metric]
        values = _nums(limit)
        if not values:
            problems.append(f"{metric}: cannot parse limit {limit!r}")
            continue
        value = values[0]
        code = float(th[key])
        if abs(value - code) > 1e-9:
            problems.append(f"{metric}: docs={value} code={code} (key={key})")
        if direction == "le" and ("≤" not in limit):
            problems.append(f"{metric}: direction should be <=, docs: {limit!r}")
        if direction == "ge" and ("≥" not in limit):
            problems.append(f"{metric}: direction should be >=, docs: {limit!r}")
        numeric += 1
        seen.add(metric)

    missing = set(expected) - seen
    if missing:
        problems.append("docs/thresholds.md missing metrics: " + ", ".join(sorted(missing)))
    if numeric != 12:
        problems.append(f"expected 12 numeric threshold rows, parsed {numeric}")
    if "p1_leak" not in seen:
        problems.append("docs/thresholds.md is missing the p1_leak rule")

    attr = _rows(md, "| 指标超限 | 归因到哪一层 |")
    names = {c[0] for c in attr if c}
    if not expected_attr <= names:
        problems.append("attribution table missing: "
                        + ", ".join(sorted(expected_attr - names)))
    for rule in ("U1", "U2", "U3"):
        if f"**{rule}**" not in md:
            problems.append(f"uncertain rule {rule} missing in docs/thresholds.md")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the offline demo and print the scorecard.")
    parser.add_argument("--out", help="where to write the demo deliverables (default: a temp dir)")
    args = parser.parse_args()

    rc = _dependency_hint()
    if rc:
        return rc

    sys.path.insert(0, str(ROOT / "scripts"))
    from cutoutlib import recipes, score  # noqa: E402
    from cutoutlib.cache import CacheMiss  # noqa: E402
    from cutoutlib.case import CaseError, load_case  # noqa: E402
    from cutoutlib.pipeline import run_case  # noqa: E402

    print("== 1. thresholds vs docs/thresholds.md ==")
    problems = check_thresholds(score)
    if problems:
        for p in problems:
            print("  mismatch: " + p)
        print("SELFCHECK: FAIL (code and docs disagree; fix one of them)")
        return 2
    print("  ok: 12 numeric thresholds + p1_leak rule match; "
          "attribution table and uncertain rules U1/U2/U3 present")

    print("== 2. offline demo ==")
    demo = ROOT / "demo"
    cache = demo / "cache" / "demo-scene" / "auto"
    required = (
        demo / "case.json", demo / "source.png", cache / "meta.json",
        cache / "bria.png", cache / "key.png",
        cache / "layers" / "z0_background.png",
        cache / "layers" / "z1_character.png",
        cache / "layers" / "z2_splash.png",
    )
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        print("  demo is incomplete, missing: " + ", ".join(missing))
        print("  regenerate with: python scripts/make_demo.py")
        print("SELFCHECK: FAIL (demo incomplete)")
        return 2

    out_dir = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="cutout-selfcheck-"))
    try:
        case = load_case(demo)
        res = run_case(case, recipes.make("auto"), out_dir=out_dir, write=True,
                       cache_root=demo / "cache")
    except CacheMiss as exc:
        print("  capture cache incomplete: " + str(exc))
        print("  (P1 = bria.png, P2 = layers/*.png, P3 = key.png; see SKILL.md)")
        print("SELFCHECK: FAIL (capture layer)")
        return 2
    except (CaseError, ValueError, OSError) as exc:
        print("  demo cannot run: " + str(exc))
        print("SELFCHECK: FAIL")
        return 2

    card = res["card"]
    m = card["metrics"]
    print("  case: " + card["case_id"])
    for key in SIX:
        print(f"  {key:<15} = {m.get(key)}")
    print(f"  alpha_area_frac = {m.get('alpha_area_frac')}   "
          f"alpha_evidence_ratio = {m.get('alpha_evidence_ratio')}   "
          f"p1_area_frac = {m.get('p1_area_frac')}   "
          f"exclude_hit = {m.get('exclude_hit')}   thin_cov = {m.get('thin_cov')}")
    print(f"  iou_ref (reference score; demo has no human reference) = {m.get('iou_ref')}")
    print(f"  status = {card['status']}   score = {card['score']}   "
          f"blocked_layer = {card.get('blocked_layer')}")
    for f in card["failures"]:
        print(f"  failed: {f['metric']} = {f['value']} (limit {f['limit']}) -> "
              f"{f.get('attribution', '')}")
    print(f"  deliverables: {out_dir}")
    print("SELFCHECK: " + ("PASS" if card["status"] == "pass" else "FAIL"))
    return 0 if card["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
