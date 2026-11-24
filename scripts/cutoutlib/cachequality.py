"""`check-cache --strict` 的质量档：P2 到底有没有给出 keep 证据。

默认档只回答「三路文件在不在」。文件齐全不等于产物有用：P2 只丢一层把全图圈进去，
或者框住了人却漏了竿与道具，文件契约一样过，融合出来的范围却会掉。

本模块只判一件事——**P2 是否覆盖了 keep 证据**，三条互不依赖的检查里至少过两条：

1. 层数：至少两层，且至少一层是「有意义的物体层」（不是底图、不是整幅图、alpha 非空）；
2. label 语义：每层都有 label，且至少一层被分到 keep 主体这一类；
3. keep 区域覆盖率：P1 的显著核心（最大连通域）被 P2 的 keep 层覆盖的比例 ≥0.80。

keep 证据取 P1 的显著核心，不取 P2 自己：用 P2 量 P2 是自证。这里只读磁盘上的产物，
不改判定逻辑，也不改任何阈值。
"""

from __future__ import annotations

import cv2
import numpy as np

from . import propose

# 一层至少要有这么多不透明像素才算「有意义的物体层」（demo 是 360x270，
# 200px 相当于全图 0.2%；再小的层对范围判断没有信息量）。
OBJECT_MIN_PX = 200
# bbox 面积达到全图的这个比例，且 label 不是底图，就当作「把全图圈进去」。
FULL_FRAME_FRACTION = 0.95
# P1 显著核心被 P2 keep 层覆盖的下限。
KEEP_COVERAGE_MIN = 0.80
KEEP_KINDS = ("subject", "thin", "fish")
BACKGROUND_WORDS = ("背景", "底图", "background", "backdrop", "base")


def _is_background(layer) -> bool:
    """底图不算「物体层」：它本来就该覆盖全图。"""
    if int(getattr(layer, "z", 0)) == 0:
        return True
    label = (getattr(layer, "label", "") or "").strip().lower()
    name = (getattr(layer, "name", "") or "").strip().lower()
    return any(word.lower() in label or word.lower() == name for word in BACKGROUND_WORDS)


def _p1_core(alpha: np.ndarray) -> np.ndarray | None:
    """P1 的显著核心：强 alpha 的最大连通域。

    取最大连通域而不是整张 alpha，是为了不被 P1 偶尔多给的大块背景带偏；
    核心为空时返回 None，由调用方判「覆盖不可测」。
    """
    strong = alpha >= 200
    if int(strong.sum()) < OBJECT_MIN_PX:
        strong = alpha >= 128
    if int(strong.sum()) < OBJECT_MIN_PX:
        return None
    n, labels, stats, _ = cv2.connectedComponentsWithStats(strong.astype(np.uint8), 8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    return labels == idx


def evaluate(prop) -> dict:
    """对一份已加载的采集缓存做 P2 质量判定，返回逐项结果与总判。"""
    shape = prop.bria.shape
    h, w = shape
    image_area = max(h * w, 1)

    rows = []
    for layer in prop.layers:
        mask = layer.alpha_on(shape)
        x0, y0, x1, y1 = layer.bbox
        bbox_area = max(x1 - x0, 0) * max(y1 - y0, 0)
        kind = propose.classify(layer.label or layer.name)
        rows.append({
            "layer": layer,
            "kind": kind,
            "mask": mask,
            "px": int((mask > 128).sum()),
            "bbox_fraction": bbox_area / image_area,
            "background": _is_background(layer),
        })

    object_rows = [r for r in rows if not r["background"]]
    informative = [r for r in object_rows
                   if r["px"] >= OBJECT_MIN_PX and r["bbox_fraction"] < FULL_FRAME_FRACTION]

    # 1. 层数：不能塌成一层，也不能只有一层把全图圈进去。
    layers_ok = len(prop.layers) >= 2 and bool(informative)
    if layers_ok:
        layers_detail = (f"{len(prop.layers)} layers total, "
                         f"{len(informative)} informative object layer(s)")
    else:
        full_frame = [r["layer"].name for r in object_rows
                      if r["bbox_fraction"] >= FULL_FRAME_FRACTION]
        bits = [f"{len(prop.layers)} layers total",
                f"{len(informative)} informative object layer(s)"]
        if full_frame:
            bits.append("full-image layer(s): " + ", ".join(full_frame))
        layers_detail = "; ".join(bits)

    # 2. label 语义：label 是融合时判断「这一层归谁」的唯一依据，不能空着。
    missing_labels = [r["layer"].name for r in rows
                      if not (r["layer"].label or "").strip()]
    keep_labelled = [r for r in informative if r["kind"] in KEEP_KINDS]
    labels_ok = not missing_labels and bool(keep_labelled)
    kinds = sorted({r["kind"] for r in informative})
    if labels_ok:
        labels_detail = "labels ok, keep-class layer(s): " + ", ".join(
            r["layer"].name for r in keep_labelled) + "; kinds: " + ",".join(kinds)
    else:
        bits = []
        if missing_labels:
            bits.append("empty label(s): " + ", ".join(missing_labels))
        if not keep_labelled:
            bits.append("no informative layer classifies as keep subject (got "
                        + (",".join(kinds) if kinds else "none") + ")")
        labels_detail = "; ".join(bits)

    # 3. keep 覆盖率：P1 显著核心有多少落在 P2 的 keep 层里。
    p2_keep = np.zeros(shape, np.uint8)
    for r in rows:
        if r["kind"] in KEEP_KINDS:
            p2_keep = np.maximum(p2_keep, r["mask"])
    core = _p1_core(prop.bria)
    if core is None:
        coverage = None
        coverage_ok = False
        coverage_detail = "cannot measure: P1 has no salient core (weak P1 too)"
    else:
        coverage = float((core & (p2_keep > 128)).sum() / max(int(core.sum()), 1))
        coverage_ok = coverage >= KEEP_COVERAGE_MIN
        coverage_detail = (f"{coverage:.4f} of P1 core covered by P2 keep layers "
                           f"(need >= {KEEP_COVERAGE_MIN:.2f})")

    checks = [
        {"id": "layers", "name": "layer count", "passed": layers_ok, "detail": layers_detail},
        {"id": "labels", "name": "label semantics", "passed": labels_ok, "detail": labels_detail},
        {"id": "keep_coverage", "name": "keep coverage", "passed": coverage_ok,
         "detail": coverage_detail},
    ]
    passed_n = sum(1 for c in checks if c["passed"])
    failed = [c["name"] for c in checks if not c["passed"]]
    passed = passed_n >= 2
    return {
        "checks": checks,
        "passed_n": passed_n,
        "passed": passed,
        "failed": failed,
        "weak_leg": None if passed else "P2",
        "p2_keep_coverage": None if coverage is None else round(coverage, 4),
        "informative_layers": [r["layer"].name for r in informative],
    }
