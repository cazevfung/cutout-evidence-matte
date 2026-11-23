"""R5.5 排除区与不连通块的纯 CV 过滤（不靠模型理解"不要水花"）。

判定只用两件事：面积、以及"与主体在 N px 内是否连通"。
丢掉的每一块都记下来，晨报单列，防止把该保留的东西悄悄扔了。
"""

from __future__ import annotations

import numpy as np

from . import imaging


def drop_small(mask: np.ndarray, min_area_px: int) -> tuple[np.ndarray, list[dict]]:
    n, lab, stats, _ = imaging.label(mask)
    keep = np.zeros_like(mask, bool)
    dropped = []
    for i in range(1, n):
        area = int(stats[i, 4])
        if area >= min_area_px:
            keep |= lab == i
        else:
            x, y, w, h, _ = stats[i]
            dropped.append({"reason": "area_below_min", "area": area, "bbox": [int(x), int(y), int(x + w), int(y + h)]})
    return keep, dropped


def drop_unconnected(mask: np.ndarray, core: np.ndarray, join_px: int = 3,
                     min_keep_frac: float = 0.0,
                     protected: np.ndarray | None = None) -> tuple[np.ndarray, list[dict]]:
    """与 core 在 join_px 内不连通的连通块整块丢弃（半透明水花、光斑、UI 碎点）。

    protected：若某个连通块本身有语义证据（例如被鱼线拴着的鱼），
    即使与主体不连通也不丢——这就是"排除项按连通性"和"保留项按证据"的分界。
    """
    n, lab, stats, _ = imaging.label(mask)
    if n <= 1:
        return mask.astype(bool), []
    near = imaging.dilate(core > 0, join_px)
    total = mask.sum()
    keep = np.zeros_like(mask, bool)
    dropped = []
    for i in range(1, n):
        comp = lab == i
        area = int(stats[i, 4])
        has_evidence = protected is not None and (comp & protected).sum() > 0.5 * area
        if (comp & near).any() or has_evidence:
            keep |= comp
        elif min_keep_frac > 0 and area >= min_keep_frac * max(total, 1):
            keep |= comp
        else:
            x, y, w, h, _ = stats[i]
            dropped.append({"reason": "disconnected_from_subject", "area": area,
                            "bbox": [int(x), int(y), int(x + w), int(y + h)]})
    return keep, dropped


def apply(mask: np.ndarray, ps_exclusions: dict, params: dict,
          protected: np.ndarray | None = None) -> tuple[np.ndarray, list[dict]]:
    """排除区优先，其次不连通，最后最小面积。返回 (mask, 丢弃记录)。

    protected：有语义证据（属于 subject 或 keep 附属物）的像素，排除区不吃它们。
    """
    dropped: list[dict] = []
    out = mask.astype(bool)
    margin = int(params.get("exclusion_margin", 2))
    if params.get("use_exclusions", True):
        for name, ex in ps_exclusions.items():
            exd = imaging.dilate(ex, margin)
            if protected is not None:
                exd = exd & ~protected
            hit = int((out & exd).sum())
            if hit:
                dropped.append({"reason": f"exclusion:{name}", "area": hit, "bbox": list(imaging.bbox_of(out & exd))})
            out &= ~exd
    core = max(imaging.components(out), key=lambda c: c.sum()) if out.any() else out
    out, d2 = drop_unconnected(out, core, int(params.get("connect_px", 3)), protected=protected)
    dropped.extend(d2)
    min_area = int(params.get("min_area_px", 60))
    out, d3 = drop_small(out, min_area)
    dropped.extend(d3)
    return out, dropped
