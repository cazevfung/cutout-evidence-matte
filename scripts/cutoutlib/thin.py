"""R6 细结构通道：由几何先验生成 1.5–2px 软边 alpha 带。

这些像素在原图里真实存在，但**语义分割拿不到**（1px 白线贴在亮水面上）。
所以只允许从原图的几何与像素统计推出，不许任何生成模型补画。
整条通道产出的像素一律标 estimated。
"""

from __future__ import annotations

import cv2
import numpy as np

from . import imaging


def band(shape: tuple[int, int], p0, p1, width: float = 1.8, soft: float = 0.6,
         extend: float = 0.0) -> np.ndarray:
    """以 p0→p1 为中轴，生成宽度 width、边缘 soft 的 float32 alpha 带。

    用"到线段的解析距离"算，不用画线再距离变换——后者在 1–2px 宽度上会有半像素偏移，
    线上取样点会掉到带子外面（实测过：中心 alpha 只有 0.83、沿线覆盖 92%）。
    extend：两端各外推多少像素，补被遮挡的竿梢/线头。
    """
    h, w = shape
    a = np.asarray(p0, np.float64)
    b = np.asarray(p1, np.float64)
    if extend > 0:
        d = b - a
        n = np.linalg.norm(d)
        if n > 1e-6:
            u = d / n
            a, b = a - u * extend, b + u * extend
    half = max(float(width), 0.5) / 2.0
    soft = max(float(soft), 1e-3)

    pad = int(np.ceil(half + soft)) + 2
    x0 = max(int(np.floor(min(a[0], b[0]))) - pad, 0)
    x1 = min(int(np.ceil(max(a[0], b[0]))) + pad + 1, w)
    y0 = max(int(np.floor(min(a[1], b[1]))) - pad, 0)
    y1 = min(int(np.ceil(max(a[1], b[1]))) + pad + 1, h)
    out = np.zeros((h, w), np.float32)
    if x1 <= x0 or y1 <= y0:
        return out

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    ab = b - a
    ab2 = float(ab @ ab)
    if ab2 < 1e-9:
        dist = np.hypot(xx - a[0], yy - a[1])
    else:
        t = ((xx - a[0]) * ab[0] + (yy - a[1]) * ab[1]) / ab2
        t = np.clip(t, 0.0, 1.0)
        px = a[0] + t * ab[0]
        py = a[1] + t * ab[1]
        dist = np.hypot(xx - px, yy - py)
    alpha = np.clip((half + soft - dist) / soft, 0.0, 1.0)
    out[y0:y1, x0:x1] = alpha.astype(np.float32)
    return out


def support(rgb: np.ndarray, p0, p1, kind: str, step: float = 1.0) -> dict:
    """沿线检查"原图在不在那里"：亮度/对比统计，判断这条带是不是真有东西。"""
    gray = np.clip(imaging.luminance(rgb), 0, 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    resp = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT if kind == "line" else cv2.MORPH_BLACKHAT, k)
    pts = imaging.sample_polyline(np.array([p0, p1], np.float64), step)
    vals = []
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < resp.shape[0] and 0 <= xi < resp.shape[1]:
            vals.append(float(resp[yi, xi]))
    vals = np.array(vals) if vals else np.zeros(1)
    return {"n": int(vals.size), "mean": float(vals.mean()),
            "frac_gt_20": float((vals > 20).mean()), "frac_gt_40": float((vals > 40).mean())}


def build(rgb: np.ndarray, shaft, params: dict | None = None) -> dict:
    params = params or {}
    width = float(params.get("thin_width", 1.8))
    soft = float(params.get("thin_soft", 0.6))
    alpha = band(rgb.shape[:2], shaft.p0, shaft.p1, width, soft,
                 extend=float(params.get("thin_extend", 2.0)))
    sup = support(rgb, shaft.p0, shaft.p1, shaft.kind,
                  step=float(params.get("support_step", 2.0)))
    return {"alpha": alpha, "support": sup, "kind": shaft.kind,
            "p0": [float(v) for v in shaft.p0], "p1": [float(v) for v in shaft.p1],
            "width": width, "estimated": True}


def coverage(alpha: np.ndarray, points: np.ndarray, thr: float = 0.5, tol: int = 1) -> float:
    """沿给定采样点，alpha≥thr 的比例（tol 像素容差）。"""
    if points.size == 0:
        return 0.0
    hard = alpha >= thr
    if tol > 0:
        hard = imaging.dilate(hard, tol)
    hit = 0
    for x, y in points:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < hard.shape[0] and 0 <= xi < hard.shape[1] and hard[yi, xi]:
            hit += 1
    return hit / len(points)
