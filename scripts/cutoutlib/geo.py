"""R4.5 几何先验：不靠语义分割，直接在原图上找竿身与鱼线。

思路：
  1. 亮细结构 / 暗细结构 = 原图的 top-hat / black-hat 响应（能把 1–2px 的细线从大面积背景里抬出来）
  2. HoughLinesP 出线段，按（角度, 距原点距离）聚类、合并共线段（鱼线常被鱼挡住断成两截）
  3. 用邻接约束筛：竿必须贴着主体；鱼线必须一端靠近竿梢
  4. 端点外推，补齐被挡掉的一小截
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np

from . import imaging


@dataclass
class Shaft:
    p0: tuple[float, float]
    p1: tuple[float, float]
    width: float
    kind: str          # "rod" | "line"
    score: float
    length: float

    def as_dict(self) -> dict:
        d = asdict(self)
        d["p0"] = [round(float(v), 2) for v in self.p0]
        d["p1"] = [round(float(v), 2) for v in self.p1]
        return d

    def polyline(self) -> np.ndarray:
        return np.array([self.p0, self.p1], np.float64)


def _segments(mask: np.ndarray, min_len: int, max_gap: int = 14) -> list[tuple[float, float, float, float]]:
    lines = cv2.HoughLinesP(mask, 1, np.pi / 720, threshold=max(30, min_len // 4),
                            minLineLength=min_len, maxLineGap=max_gap)
    if lines is None:
        return []
    return [tuple(float(v) for v in l[0]) for l in lines]


def _theta_rho(seg) -> tuple[float, float]:
    x0, y0, x1, y1 = seg
    dx, dy = x1 - x0, y1 - y0
    theta = np.degrees(np.arctan2(dy, dx)) % 180.0
    n = np.hypot(dx, dy) or 1.0
    rho = abs(-dy / n * x0 + dx / n * y0)
    return theta, rho


def _merge_collinear(segs: list, dtheta=3.0, drho=5.0) -> list:
    groups: list[list] = []
    for s in segs:
        th, rho = _theta_rho(s)
        placed = False
        for g in groups:
            gth, grho = _theta_rho(g[0])
            if min(abs(th - gth), 180 - abs(th - gth)) <= dtheta and abs(rho - grho) <= drho:
                g.append(s)
                placed = True
                break
        if not placed:
            groups.append([s])
    merged = []
    for g in groups:
        pts = []
        for s in g:
            pts.append((s[0], s[1]))
            pts.append((s[2], s[3]))
        pts = np.array(pts, np.float64)
        d = pts[-1] - pts[0]
        proj = (pts - pts[0]) @ d / (np.linalg.norm(d) ** 2 + 1e-9)
        merged.append(tuple(np.r_[pts[proj.argmin()], pts[proj.argmax()]]))
    return merged


def _width_along(ridge: np.ndarray, seg) -> float:
    """沿线的半宽：用响应带的距离变换取中位数，宽度 = 2 * 到带边的距离。"""
    d = cv2.distanceTransform((ridge > 0).astype(np.uint8), cv2.DIST_L2, 5)
    x0, y0, x1, y1 = seg
    n = max(int(np.hypot(x1 - x0, y1 - y0)), 1)
    vals = []
    for i in range(0, n + 1, max(1, n // 80)):
        t = i / n
        xi = int(round(x0 + (x1 - x0) * t))
        yi = int(round(y0 + (y1 - y0) * t))
        if 0 <= yi < d.shape[0] and 0 <= xi < d.shape[1]:
            vals.append(d[yi, xi])
    if not vals:
        return 0.0
    return float(max(1.0, 2.0 * float(np.median(vals))))


def _dist_to_mask(mask: np.ndarray, pt) -> float:
    d = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    x, y = int(round(pt[0])), int(round(pt[1]))
    x = min(max(x, 0), mask.shape[1] - 1)
    y = min(max(y, 0), mask.shape[0] - 1)
    return float(d[y, x])


def _inside_points(shaft, mask: np.ndarray, grow: int = 2, step: float = 2.0) -> float:
    """沿线采样，返回落在 mask（外扩 grow）里的比例。"""
    m = imaging.dilate(mask > 0, int(grow))
    pts = imaging.sample_polyline(np.array([shaft[0], shaft[1]], np.float64), step)
    if not len(pts):
        return 0.0
    hit = 0
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < m.shape[0] and 0 <= xi < m.shape[1] and m[yi, xi]:
            hit += 1
    return hit / len(pts)


def find_shafts(rgb: np.ndarray, subject_mask: np.ndarray, params: dict | None = None,
                nonkeep: np.ndarray | None = None) -> list[Shaft]:
    """找细长结构。

    两条通用门槛（CUT-003 返工2），不看题材：
    A. **邻接证据**：至少要有一个端点真正落在 keep 主体里（外扩 2px），
       "离主体 60px 以内"不算 —— 建筑的长直边常常正好从角色旁边经过，按距离会误收。
    B. **不是纯背景**：线段沿途有 ≥80% 落在 P2 判定为"非 keep"的图层里 → 一律丢弃。
    """
    params = params or {}
    h, w = subject_mask.shape
    gray = np.clip(imaging.luminance(rgb), 0, 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(params.get("tophat_k", 9)),) * 2)
    bright = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)

    min_len = int(params.get("min_len_frac", 0.06) * max(h, w))
    out: list[Shaft] = []

    for name, resp, thr in (("line", bright, params.get("bright_thr", 40)),
                            ("rod", dark, params.get("dark_thr", 35))):
        ridge = (resp > thr).astype(np.uint8) * 255
        segs = _segments(ridge, min_len, int(params.get("max_gap", 16)))
        if not segs:
            continue
        segs = _merge_collinear(segs, params.get("dtheta", 3.0), params.get("drho", 6.0))
        for s in segs:
            length = float(np.hypot(s[2] - s[0], s[3] - s[1]))
            if length < min_len:
                continue
            width = _width_along(ridge, s)
            p0, p1 = (s[0], s[1]), (s[2], s[3])
            d0 = _dist_to_mask(subject_mask, p0)
            d1 = _dist_to_mask(subject_mask, p1)
            near = min(d0, d1)
            # 门槛 A：至少一端真的在主体里（不是"靠近"）
            if near > params.get("endpoint_inside_px", 2.0):
                continue
            # 门槛 B：不能是一条纯背景的长直边
            if nonkeep is not None and nonkeep.any():
                frac = _inside_points((p0, p1), nonkeep, grow=0, step=2.0)
                if frac >= params.get("nonkeep_reject_frac", 0.8):
                    continue
            score = length / (1.0 + width) * (1.0 if near < 8 else 0.85)
            out.append(Shaft(p0=p0, p1=p1, width=width, kind=name, score=score, length=length))
    out.sort(key=lambda s: -s.score)

    if params.get("extend_line", True):
        out = _extend(out, subject_mask, params)
    return out


def _extend(shafts: list[Shaft], subject_mask: np.ndarray, params: dict) -> list[Shaft]:
    """端点外推：沿线段方向往外再多找一小段（≤ 最大外推像素），补被挡住的竿梢。"""
    max_ext = float(params.get("max_extend_px", 40))
    for s in shafts:
        if s.kind != "rod":
            continue
        # 竿梢那一端：离主体更远的一端，往同方向外推
        d0 = _dist_to_mask(subject_mask, s.p0)
        d1 = _dist_to_mask(subject_mask, s.p1)
        far, near = (s.p0, s.p1) if d0 >= d1 else (s.p1, s.p0)
        v = np.array(far, float) - np.array(near, float)
        nv = np.linalg.norm(v)
        if nv < 1e-6:
            continue
        v = v / nv
        # 外推量受"已经离开主体多远"约束，避免凭空加长
        ext = min(max_ext, max(0.0, d0 if d0 >= d1 else d1))
        new_far = tuple(np.array(far, float) + v * ext)
        if d0 >= d1:
            s.p0 = new_far
        else:
            s.p1 = new_far
        s.length = float(np.linalg.norm(np.array(s.p1, float) - np.array(s.p0, float)))
    return shafts


def rod_and_line(shafts: list[Shaft], subject_mask: np.ndarray) -> dict:
    """把检测到的线段组织成「竿 + 鱼线」的拓扑，给 continuity 与 R6 用。"""
    rods = [s for s in shafts if s.kind == "rod"]
    lines = [s for s in shafts if s.kind == "line"]
    rod = rods[0] if rods else None
    if rod is None:
        # 没有竿也要能工作：细长目标可能只检出一条线（风筝线、旗杆、天线）
        line = lines[0] if lines else None
        return {"rod": None, "line": line}
    # 鱼线的哪一端离竿的哪一端最近 → 那一端就是竿梢（物理约束，比"离主体远近"可靠）
    line = lines[0] if lines else None
    if line is not None:
        d_tip = min(np.hypot(line.p0[0] - rod.p0[0], line.p0[1] - rod.p0[1]),
                    np.hypot(line.p1[0] - rod.p0[0], line.p1[1] - rod.p0[1]))
        d_butt = min(np.hypot(line.p0[0] - rod.p1[0], line.p0[1] - rod.p1[1]),
                     np.hypot(line.p1[0] - rod.p1[0], line.p1[1] - rod.p1[1]))
        tip, grip = (rod.p0, rod.p1) if d_tip <= d_butt else (rod.p1, rod.p0)
    else:
        d0 = _dist_to_mask(subject_mask, rod.p0)
        d1 = _dist_to_mask(subject_mask, rod.p1)
        tip, grip = (rod.p0, rod.p1) if d0 > d1 else (rod.p1, rod.p0)
    return {"rod": rod, "line": line, "tip": tip, "grip": grip}


def path_points(rod_line: dict, domain_shape) -> np.ndarray:
    """竿顶 → 竿梢 → 线末端 的采样点（continuity / R6 共用）。只有线时就用线本身。"""
    rod = rod_line.get("rod")
    if rod is None:
        line = rod_line.get("line")
        if line is None:
            return np.zeros((0, 2))
        return imaging.sample_polyline(np.array([line.p0, line.p1], np.float64), 1.0)
    tip = rod_line["tip"]
    grip = rod_line["grip"]
    pts = [grip, tip]
    line = rod_line.get("line")
    if line is not None:
        ends = [line.p0, line.p1]
        near = min(ends, key=lambda p: np.hypot(p[0] - tip[0], p[1] - tip[1]))
        far = ends[0] if near is ends[1] else ends[1]
        pts.extend([near, far])
    poly = np.asarray(pts, np.float64)
    keep = [(p[0] >= 0 and p[1] >= 0 and p[0] < domain_shape[1] and p[1] < domain_shape[0]) for p in poly]
    if all(keep):
        return imaging.sample_polyline(poly, 1.0)
    return imaging.sample_polyline(poly[keep], 1.0) if any(keep) else np.zeros((0, 2))


def prepare(mask: np.ndarray, structural: bool = False) -> np.ndarray:
    """给掩膜做 3x3 开闭，去掉 1px 毛刺（做距离变换前用）。"""
    m = mask.astype(np.uint8)
    if not structural:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m > 0
