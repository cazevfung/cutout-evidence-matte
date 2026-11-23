"""背景色模型（CUT-004 返工）：从"明确不是主体"的地方取色，用来否决漏进 alpha 的环境像素。

取色来源三处，互相补位：

1. **P2 的非 keep 层**（背景/道具/雕塑/房屋那些图层）—— 语义上最干净的背景样本；
2. **图像四边框带**（默认 24px）—— 覆盖"没有任何图层描述"的背景，例如地面、墙面延伸出去的部分；
3. 两者的交集都排除掉 **keep 主体外扩 2px** 的像素，避免把角色自己的颜色算进背景。

模型是 Lab 空间里的一小组聚类中心（默认 12 个）。判据统一用 ΔE（CIE76）：
`min_c ||Lab(p) - centroid_c||`，阈值见 `BACKGROUND_TOL`，标定依据写在
`reports/case-06-07-缝隙与水体.md`。
"""

from __future__ import annotations

import cv2
import numpy as np

from . import imaging

#: 洞内颜色与背景模型的距离在这个值以内 → 认为"洞内就是背景"，不填。
#: 标定：case-07 的水体洞 ΔE 1.3–1.9、case-06 的地面洞 3.2–4.6；case-01 角色自身的
#: 闭合缝 ΔE 6.4–8.8（那几处是角色自己的颜色，应当照填）。5.0 卡在两组中间。
HOLE_TOL = 5.0
#: 逐像素否决用的阈值：比洞判定更严，只吃"几乎就是背景色"的像素。
VM_TOL = 3.0
#: 洞小于这个面积一律照旧填掉 —— 免得把贴图噪点留成针孔。
HOLE_MIN_AREA = 200
#: 否决区域的最小面积与同质性要求（防止把围巾上的白色高光当成背景挖掉）
VETO_MIN_AREA = 400
VETO_P90_TOL = 4.0


def to_lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)


def border_mask(shape: tuple[int, int], px: int = 24) -> np.ndarray:
    h, w = shape
    m = np.zeros(shape, bool)
    m[:px, :] = m[-px:, :] = m[:, :px] = m[:, -px:] = True
    return m


def sample(ps, shape: tuple[int, int], px: int = 24) -> dict:
    """返回背景样本掩膜与统计（供报告引用）。

    三个来源：
    - 被 P2 明确归为非 keep 的图层（水花/道具/背景物件）；
    - 画面四周 24px 带（去掉主体邻域）；
    - **图层拆分说"这里没有任何物体"的全部像素**（= 非 base 图层以外的全图）。
      这一条是必需的：只靠边框带采不到被主体围在中间的背景物件（桌面、底座、平台），
      它们的颜色不在调色板里，就会被当成"异色"，把主体围出的真实缝隙当成漏抠填上。
    """
    keep = ps.p2 > 128
    for att in ps.attachments.values():
        keep |= att > 128
    nonkeep = ps.p2_nonkeep if ps.p2_nonkeep is not None else np.zeros(shape, bool)
    grow = imaging.dilate(keep, 2)
    from_layers = nonkeep & ~grow
    from_border = border_mask(shape, px) & ~imaging.dilate(keep, 4)
    from_base = np.zeros(shape, bool)
    fg = getattr(ps, "p2_foreground", None)
    if fg is not None and fg.any():
        from_base = ~imaging.dilate(fg >= 40, 4)
    mask = from_layers | from_border | from_base
    return {"mask": mask, "from_layers": int(from_layers.sum()),
            "from_border": int(from_border.sum()), "from_base": int(from_base.sum()),
            "keep": int(keep.sum())}


def palette(lab: np.ndarray, mask: np.ndarray, k: int = 12, seed: int = 7) -> np.ndarray:
    px = lab[mask]
    if len(px) == 0:
        return np.zeros((1, 3), np.float32)
    if len(px) > 20000:
        px = px[np.random.default_rng(seed).choice(len(px), 20000, replace=False)]
    k = int(min(k, len(px)))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, centers = cv2.kmeans(px.reshape(-1, 3).astype(np.float32), k, None, crit, 3,
                               cv2.KMEANS_PP_CENTERS)
    return centers


def model_stats(lab: np.ndarray, mask: np.ndarray, dist: np.ndarray, centers: np.ndarray) -> dict:
    """背景样本自身的 ΔE 分布（用于 C1-c 的"防自证"判据：比模型标准差还近 = 基本就是背景色）。"""
    d = dist[mask]
    if len(d) == 0:
        return {"samples": 0, "mean": 0.0, "std": 0.0, "p90": 0.0}
    return {"samples": int(len(d)), "mean": round(float(d.mean()), 3),
            "std": round(float(d.std()), 3), "p90": round(float(np.percentile(d, 90)), 3)}


def distance(lab: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """每个像素到最近背景中心的 ΔE。分块算，避免 (H*W, K, 3) 撑爆内存。"""
    flat = lab.reshape(-1, 3)
    out = np.empty(len(flat), np.float32)
    step = 200000
    for i in range(0, len(flat), step):
        blk = flat[i:i + step]
        d = np.linalg.norm(blk[:, None, :] - centers[None, :, :], axis=2)
        out[i:i + step] = d.min(axis=1)
    return out.reshape(lab.shape[:2])


def fill_holes_by_color(mask: np.ndarray, lab: np.ndarray, dist: np.ndarray,
                        tol: float = HOLE_TOL, min_area: int = HOLE_MIN_AREA,
                        max_fill_px: int = 0) -> tuple[np.ndarray, list]:
    """填洞，但先看洞里的颜色像不像背景：像背景说明那是"围出来的真实缝隙"，不填。

    max_fill_px：大到这个面积的洞一律不填。填洞只用来修分割在实心区域留下的小孔；
    能装下另一个物体的洞是"看穿"的缝隙，里面可能还有别的背景物件（底座、支架、台面），
    填上等于把那些物件贴到主体上（case-09 两腿之间的 4.7 万像素就是这样被填的）。

    返回 (填好洞的 mask, 决策记录)。
    """
    holes = imaging.fill_holes(mask) & ~mask
    if not holes.any():
        return mask.astype(bool), []
    n, lab_idx, stats, _ = imaging.label(holes)
    out = mask.astype(bool).copy()
    log = []
    for i in range(1, n):
        comp = lab_idx == i
        area = int(stats[i, 4])
        med = float(np.median(dist[comp]))
        if area < min_area:
            out |= comp
            log.append({"area": area, "median_dE": round(med, 2), "decision": "fill(small)"})
            continue
        if max_fill_px and area > max_fill_px:
            log.append({"area": area, "median_dE": round(med, 2), "decision": "keep-open(large-gap)"})
            continue
        if med <= tol:
            log.append({"area": area, "median_dE": round(med, 2), "decision": "keep-open(background)"})
        else:
            out |= comp
            log.append({"area": area, "median_dE": round(med, 2), "decision": "fill(different-color)"})
    return out, log


def veto_background(alpha: np.ndarray, dist: np.ndarray, band_px: int = 2,
                    tol: float = VM_TOL, min_area: int = VETO_MIN_AREA,
                    p90_tol: float = VETO_P90_TOL,
                    protect: np.ndarray | None = None) -> tuple[np.ndarray, list]:
    """把"颜色就是背景、又不在轮廓软边带里"的成片像素从 alpha 里剔掉。

    只吃成片的（≥min_area）、同质的（p90 ≤ p90_tol）区域：
    围巾高光、眼镜反光这类小面积近背景色不会被挖掉。

    protect：图层拆分已经认作主体的像素。白墙前的白蕾丝、浅金发丝在 Lab 里与白墙几乎同色，
    不允许只凭颜色把它冠掉 —— 漏进的背景本来就不在主体层里，这条不会放走它。
    """
    solid = alpha > 0.5
    if not solid.any():
        return alpha, []
    band = imaging.dilate(solid, band_px) & ~imaging.erode(solid, band_px)
    cand = solid & ~band & (dist <= tol)
    if protect is not None:
        cand &= ~protect
    if not cand.any():
        return alpha, []
    n, lab_idx, stats, _ = imaging.label(cand)
    out = alpha.copy()
    log = []
    for i in range(1, n):
        comp = lab_idx == i
        area = int(stats[i, 4])
        if area < min_area:
            continue
        d = dist[comp]
        if float(np.percentile(d, 90)) > p90_tol:
            log.append({"area": area, "p90_dE": round(float(np.percentile(d, 90)), 2),
                        "decision": "keep(heterogeneous)"})
            continue
        out[comp] = 0.0
        ys, xs = np.nonzero(comp)
        log.append({"area": area, "p90_dE": round(float(np.percentile(d, 90)), 2),
                    "decision": "veto", "bbox": [int(xs.min()), int(ys.min()),
                                                 int(xs.max()), int(ys.max())]})
    return out, log
