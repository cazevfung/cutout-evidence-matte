"""配方空间与变异（R8）。

配方只描述"决策层怎么算"，不描述"调哪次云"——所以换配方免费、可整夜跑。
每个配方都带随机种子，保证 AC09 的重跑可复现。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

AUTO = {
    "name": "auto",
    "seed": 7,
    "p1_thr": 128,
    "use_p2": True,
    "p2_thr": 128,
    "p2_join_px": 3,
    "p2_min_area": 40,
    "p2_authority": True,
    "p2_authority_px": 4,
    "use_p3": False,
    "p3_join_px": 6,
    "p3_min_area": 200,
    "fill_holes": True,
    "keep_attachments": True,
    "attach_join_px": 6,
    "attach_max_frac": 0.05,
    "use_exclusions": True,
    "exclusion_margin": 2,
    "min_area_frac": 0.0005,
    "min_rel_area": 0.02,
    "connect_px": 3,
    "min_area_px": 60,
    "matte_on": True,
    "trimap_erode": 2,
    "trimap_dilate": 1,
    "fb_radius": 9,
    "min_fb_energy": 900.0,
    "geo_soft": 3.0,
    "thin_on": True,
    "thin_width": 1.8,
    "thin_soft": 0.6,
    "bright_thr": 40,
    "dark_thr": 35,
    "tophat_k": 9,
    "min_len_frac": 0.06,
    "max_gap": 16,
    "rod_join_px": 24,
    "line_join_px": 60,
    "max_extend_px": 40,
    "support_step": 2.0,
    "alpha_floor": 0.0,
    "hole_color_check": True,
    "hole_tol": 5.0,
    "hole_min_area": 200,
    "hole_max_fill_frac": 0.001,
    "bg_veto": True,
    "bg_k": 12,
    "bg_veto_tol": 2.5,
    "veto_band_px": 2,
    "veto_min_area": 800,
    "veto_p90_tol": 3.5,
    "close_r": 1,
}

SPACE = [
    ("auto", {}),
    ("p1-skeleton", {"p2_authority": False}),
    ("no-thin", {"thin_on": False}),
    ("no-matte", {"matte_on": False}),
    ("tight-excl", {"exclusion_margin": 4}),
    ("loose-excl", {"exclusion_margin": 0}),
    ("use-p3", {"use_p3": True}),
    ("no-attach", {"keep_attachments": False}),
    ("wide-trimap", {"trimap_dilate": 5, "trimap_erode": 1}),
    ("narrow-trimap", {"trimap_dilate": 2, "trimap_erode": 3}),
    ("dense-join", {"p2_join_px": 6, "connect_px": 5}),
]


def make(name: str, overrides: dict | None = None) -> dict:
    r = copy.deepcopy(AUTO)
    r["name"] = name
    if overrides:
        r.update(overrides)
    return r


def space() -> list[dict]:
    return [make(n, o) for n, o in SPACE]


def mutate(recipe: dict, metric: str, seed: int) -> dict:
    """按归因做一步变异（免费）。"""
    r = copy.deepcopy(recipe)
    r["seed"] = seed
    r["name"] = f"{recipe['name']}+{metric}"
    if metric == "halo":
        r["min_fb_energy"] = min(r["min_fb_energy"] + 300.0, 3600.0)
        r["trimap_dilate"] = max(r["trimap_dilate"] - 1, 1)
    elif metric == "stray":
        r["min_area_px"] = int(r["min_area_px"] * 1.6)
        r["p2_min_area"] = int(r["p2_min_area"] * 1.4)
        r["connect_px"] = r["connect_px"] + 1
    elif metric == "hole":
        r["fill_holes"] = True
        r["trimap_erode"] = max(r["trimap_erode"] - 1, 0)
    elif metric == "coverage":
        r["p2_join_px"] = r["p2_join_px"] + 2
        r["use_p3"] = True
        r["attach_join_px"] = r["attach_join_px"] + 2
    elif metric == "continuity":
        r["thin_on"] = True
        r["thin_width"] = min(r["thin_width"] + 0.4, 2.4)
        r["min_len_frac"] = max(r["min_len_frac"] - 0.02, 0.02)
    elif metric == "exclude_hit":
        r["use_exclusions"] = True
        r["exclusion_margin"] = r["exclusion_margin"] + 1
    elif metric == "iou_ref":
        r["exclusion_margin"] = max(r["exclusion_margin"] - 1, 0)
        r["trimap_dilate"] = r["trimap_dilate"] + 1
    return r


def save(recipe: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(recipe, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    return p


def load(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"配方不存在：{p.resolve()}")
    data = json.loads(p.read_text(encoding="utf-8"))
    merged = copy.deepcopy(AUTO)
    merged.update(data)
    return merged
